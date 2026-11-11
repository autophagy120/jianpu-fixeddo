"""Pull UTF-8 and UTF-16LE string literals out of Cython-built .pyd modules."""
import os
import re
import sys

UTF8_RUN = re.compile(rb"[\x20-\x7e\xc2-\xf4][\x20-\x7e\x80-\xbf]*")
UTF16_RUN = re.compile(rb"(?:[\x20-\x7e]\x00|[\x00-\xff][\x4e-\x9f]){4,}")


def utf8_strings(data, minlen):
    out = []
    for m in UTF8_RUN.finditer(data):
        try:
            s = m.group().decode("utf8")
        except UnicodeDecodeError:
            continue
        if len(s) >= minlen and not s.startswith("__pyx_"):
            out.append(s)
    return out


def utf16_strings(data, minlen):
    out = []
    for m in UTF16_RUN.finditer(data):
        try:
            s = m.group().decode("utf-16-le")
        except UnicodeDecodeError:
            continue
        if len(s) >= minlen:
            out.append(s)
    return out


def main(path, outpath, minlen=4):
    data = open(path, "rb").read()
    a = utf8_strings(data, minlen)
    b = utf16_strings(data, minlen)
    with open(outpath, "w", encoding="utf8") as fh:
        fh.write("=== UTF8 ===\n")
        for s in a:
            if any(ord(c) > 127 for c in s):
                fh.write(s + "\n")
        fh.write("\n=== UTF16 ===\n")
        for s in b:
            fh.write(s + "\n")
    print(path, "utf8:", len(a), "utf16:", len(b))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
