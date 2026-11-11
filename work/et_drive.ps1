# easytr (简谱转调工具) 自动化驱动脚本
#
# 背景：这个 exe 是 PyInstaller + Cython(32位 py36)，无法 import；只能驱动 GUI。
# 已验证：
#   - 直接启动会报 "could not find or load the Qt platform plugin windows"
#     → 必须设 QT_QPA_PLATFORM_PLUGIN_PATH / QT_PLUGIN_PATH（本脚本已设）
#   - UIAutomation 枚举到 0 个控件（PyQt 未开无障碍桥，QT_ACCESSIBILITY=1 也无效）
#   - PostMessage(WM_LBUTTONDOWN/UP) 点击无效（Qt 忽略合成消息，截图字节不变）
#     → 必须用 SendInput 真实输入（会移动物理光标）
#   - PrintWindow(PW_RENDERFULLCONTENT) **可以**抓窗口内容，不受前台/桌面会话影响
#
# 界面坐标（窗口被 resize 到 760x560 后实测，窗口内相对坐标）：
#   文件菜单   (40, 65)
#   原调下拉   (140,110)      转调下拉 (320,110)
#   识别按钮   (410,110)      修正按钮 (465,110)
#   谱面显示区 (280,420)
#
# 用法：
#   .\et_drive.ps1 -Action launch          启动并设置好 Qt 环境
#   .\et_drive.ps1 -Action capture         抓当前窗口到 work\et_win.png
#   .\et_drive.ps1 -Action click -X 410 -Y 110
#   .\et_drive.ps1 -Action type  -Text "C:\path\to.jpg"
#   .\et_drive.ps1 -Action key   -Key 13    # 13=Enter 27=Esc 9=Tab 40=Down
#   .\et_drive.ps1 -Action kill

param(
  [string]$Action = "capture",
  [int]$X = 0,
  [int]$Y = 0,
  [string]$Text = "",
  [int]$Key = 13
)

$ErrorActionPreference = "Stop"
$Root = "D:\简谱转调项目"
$AppDir = Join-Path $Root "work\extracted\JianPu-Easy-Trans"
$Exe = Join-Path $AppDir "Jianpu Easy Trans.exe"
$OutPng = Join-Path $Root "work\et_win.png"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Et {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x,int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f,uint dx,uint dy,uint d,IntPtr e);
  [DllImport("user32.dll")] public static extern void keybd_event(byte vk,byte scan,uint f,IntPtr e);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h,IntPtr hdc,uint flags);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h,out RECT r);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h,IntPtr after,int x,int y,int cx,int cy,uint flags);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [StructLayout(LayoutKind.Sequential)] public struct RECT{ public int L,T,R,B; }
}
"@
Add-Type -AssemblyName System.Drawing

function Get-App() {
  return Get-Process | Where-Object { $_.ProcessName -like "*Jianpu*" } | Select-Object -First 1
}

function Launch-App() {
  $env:QT_QPA_PLATFORM_PLUGIN_PATH = Join-Path $AppDir "PyQt5\Qt\plugins\platforms"
  $env:QT_PLUGIN_PATH = Join-Path $AppDir "PyQt5\Qt\plugins"
  $p = Start-Process -FilePath $Exe -PassThru
  Start-Sleep -Seconds 10
  $p.Refresh()
  [Et]::SetWindowPos($p.MainWindowHandle, [IntPtr]::Zero, 50, 50, 760, 560, 0x0040) | Out-Null
  Start-Sleep -Seconds 2
  Write-Host "已启动 PID=$($p.Id)"
}

function Save-Capture() {
  $p = Get-App
  if (-not $p) { throw "进程不在" }
  $h = $p.MainWindowHandle
  $r = New-Object Et+RECT
  [Et]::GetWindowRect($h, [ref]$r) | Out-Null
  $w = $r.R - $r.L; $ht = $r.B - $r.T
  $bmp = New-Object System.Drawing.Bitmap $w, $ht
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $hdc = $g.GetHdc()
  [Et]::PrintWindow($h, $hdc, 2) | Out-Null
  $g.ReleaseHdc($hdc)
  $bmp.Save($OutPng, [System.Drawing.Imaging.ImageFormat]::Png)
  Write-Host "captured -> $OutPng  (${w}x${ht})"
}

function Click-At([int]$rx, [int]$ry) {
  $p = Get-App
  if (-not $p) { throw "进程不在" }
  $h = $p.MainWindowHandle
  [Et]::SetForegroundWindow($h) | Out-Null
  Start-Sleep -Milliseconds 300
  $r = New-Object Et+RECT
  [Et]::GetWindowRect($h, [ref]$r) | Out-Null
  [Et]::SetCursorPos(($r.L + $rx), ($r.T + $ry)) | Out-Null
  Start-Sleep -Milliseconds 150
  [Et]::mouse_event(0x0002, 0, 0, 0, [IntPtr]::Zero)   # left down
  Start-Sleep -Milliseconds 80
  [Et]::mouse_event(0x0004, 0, 0, 0, [IntPtr]::Zero)   # left up
  Start-Sleep -Milliseconds 400
  Write-Host "clicked ($rx,$ry)"
}

function Type-Text([string]$s) {
  # 走剪贴板 + Ctrl+V 比逐键可靠（路径含中文/空格）
  Set-Clipboard -Value $s
  Start-Sleep -Milliseconds 200
  [Et]::keybd_event(0x11, 0, 0, [IntPtr]::Zero)        # Ctrl down
  [Et]::keybd_event(0x56, 0, 0, [IntPtr]::Zero)        # V
  [Et]::keybd_event(0x56, 0, 2, [IntPtr]::Zero)
  [Et]::keybd_event(0x11, 0, 2, [IntPtr]::Zero)
  Start-Sleep -Milliseconds 500
  Write-Host "typed: $s"
}

function Send-Key([int]$vk) {
  [Et]::keybd_event([byte]$vk, 0, 0, [IntPtr]::Zero)
  Start-Sleep -Milliseconds 80
  [Et]::keybd_event([byte]$vk, 0, 2, [IntPtr]::Zero)
  Start-Sleep -Milliseconds 400
  Write-Host "key $vk"
}

switch ($Action) {
  "launch"  { Launch-App }
  "capture" { Save-Capture }
  "click"   { Click-At $X $Y }
  "type"    { Type-Text $Text }
  "key"     { Send-Key $Key }
  "kill"    { Get-App | Stop-Process -Force -ErrorAction SilentlyContinue; Write-Host "killed" }
  default   { Write-Host "unknown action: $Action" }
}
