"""Unpack a PyInstaller 3.x PYZ archive (out00-PYZ.pyz) into .pyc files.

The PYZ entries are zlib-compressed marshal byte streams written by the Python
that built the app (here 3.6), so we do NOT re-marshal them: each entry is
written out as a synthetic .pyc with the PYZ's own magic + zero mtime/size.
"""
import os
import struct
import sys
import zlib


def main(src, outdir):
    data = open(src, "rb").read()
    assert data[:4] == b"PYZ\0", data[:8]
    pyc_magic = data[4:8]
    toc_pos = struct.unpack("!i", data[8:12])[0]
    import marshal

    toc = marshal.loads(data[toc_pos:])
    if isinstance(toc, dict):
        toc = list(toc.items())
    total = 0
    names = []
    for name, entry in toc:
        typ, pos, length = entry
        blob = zlib.decompress(data[pos:pos + length])
        out = os.path.join(outdir, *name.split(".")) + ".pyc"
        os.makedirs(os.path.dirname(out) or outdir, exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(pyc_magic + b"\x00" * 8 + blob)
        total += 1
        names.append(name)
    print("pyc magic:", pyc_magic.hex())
    print("total modules:", total)
    print("\n".join(names))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
