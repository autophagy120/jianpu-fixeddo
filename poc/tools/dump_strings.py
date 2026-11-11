"""Dump printable-string runs from a Cython .pyd, decoding CJK (utf-8) properly."""
import re
import sys

PRINT = set(range(0x20, 0x7F))


def runs(data, minlen=5):
    out = []
    cur = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        ok = b in PRINT
        if not ok and 0xC2 <= b <= 0xF4:  # potential utf-8 lead
            # validate sequence
            if b < 0xE0:
                need = 1
            elif b < 0xF0:
                need = 2
            else:
                need = 3
            seq = data[i:i + 1 + need]
            if len(seq) == 1 + need and all(0x80 <= c <= 0xBF for c in seq[1:]):
                try:
                    seq.decode("utf8")
                    ok = True
                except UnicodeDecodeError:
                    ok = False
            if ok:
                cur.extend(seq)
                i += 1 + need
                continue
        if ok:
            cur.append(b)
            i += 1
        else:
            if len(cur) >= minlen:
                out.append(bytes(cur))
            cur = bytearray()
            i += 1
    if len(cur) >= minlen:
        out.append(bytes(cur))
    return out


def main(path, outpath, minlen=5):
    data = open(path, "rb").read()
    rs = runs(data, minlen)
    with open(outpath, "w", encoding="utf8") as fh:
        for r in rs:
            try:
                s = r.decode("utf8")
            except UnicodeDecodeError:
                s = r.decode("latin1")
            # drop obviously binary-ish runs
            if sum(ch.isprintable() for ch in s) < len(s) * 0.95:
                continue
            fh.write(s + "\n")
    print(path, len(rs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
