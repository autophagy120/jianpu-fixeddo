"""Minimal PyInstaller CArchive + PYZ extractor (PyInstaller 3.x layout).

Usage: python extract_pyinstaller.py <exe> <outdir>
"""
import os
import struct
import sys
import zlib

MAGIC = b"MEI\x0c\x0b\x0a\x0b"


def find_cookie(data):
    idx = data.find(MAGIC)
    last = -1
    while idx != -1:
        last = idx
        idx = data.find(MAGIC, idx + 1)
    return last


def main(exe, outdir):
    with open(exe, "rb") as fh:
        data = fh.read()
    pos = find_cookie(data)
    if pos < 0:
        raise SystemExit("no PyInstaller cookie found")
    magic, pkglen, tocoff, toclen, pyver = struct.unpack("!8sIIII", data[pos:pos + 24])
    pylib = data[pos + 24:pos + 88].rstrip(b"\x00").decode("utf8", "replace")
    print(f"cookie at {pos} pyver={pyver} pylib={pylib!r} pkglen={pkglen} tocoff={tocoff} toclen={toclen}")
    # toc offset is relative to the archive start
    arc_start = pos + 88 - pkglen
    os.makedirs(outdir, exist_ok=True)
    toc = data[arc_start + tocoff:arc_start + tocoff + toclen]
    n = len(toc)
    off = 0
    entries = []
    while off < n:
        (elen,) = struct.unpack("!i", toc[off:off + 4])
        if elen <= 0:
            break
        (epos,) = struct.unpack("!i", toc[off + 4:off + 8])
        (csize,) = struct.unpack("!i", toc[off + 8:off + 12])
        (usize,) = struct.unpack("!i", toc[off + 12:off + 16])
        flag = toc[off + 16]
        typ = chr(toc[off + 17]) if isinstance(toc[off + 17], int) else toc[off + 17:off + 18].decode()
        name = toc[off + 18:off + elen].rstrip(b"\x00").decode("utf8", "replace")
        entries.append((elen, epos, csize, usize, flag, typ, name))
        off += elen
    print(f"{len(entries)} toc entries")
    for elen, epos, csize, usize, flag, typ, name in entries:
        blob = data[arc_start + epos:arc_start + epos + csize]
        if flag == 1:
            try:
                blob = zlib.decompress(blob)
            except zlib.error as exc:
                print("  !! decompress failed", name, exc)
                continue
        out = os.path.join(outdir, name.replace("/", os.sep))
        os.makedirs(os.path.dirname(out) or outdir, exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(blob)
        print(f"  {typ} {name} {csize} -> {len(blob)}")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
