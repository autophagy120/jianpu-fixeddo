@echo off
cd /d "%~dp0extracted\JianPu-Easy-Trans"
set QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0extracted\JianPu-Easy-Trans\PyQt5\Qt\plugins\platforms
set QT_PLUGIN_PATH=%~dp0extracted\JianPu-Easy-Trans\PyQt5\Qt\plugins
start "" "Jianpu Easy Trans.exe"
