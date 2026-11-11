"""Filter dumped .pyd strings down to the ones that actually describe the program."""
import re
import sys

CJK = re.compile("[\u4e00-\u9fff]")
IDENT = re.compile(r"^[A-Za-z_%][A-Za-z0-9_ ().,%\[\]:;+*/<>='\"-]*$")
NOISE = re.compile(
    r"^(Py[A-Z]|_?(?:Jv|gcc|dll|init|fini)|__|libgcc|GCC:|MSVC|Mingw|cython|_cython_)"
)


def keep(s):
    if len(CJK.findall(s)) >= 2:
        return True
    if s.startswith("libgcc") or "GCC:" in s:
        return False
    if IDENT.match(s) and len(s) >= 9 and ("_" in s or "%" in s or "(" in s):
        return True
    if ("%" in s or "{}" in s) and " " in s and len(s) > 12:
        return True
    return False


def main(path, outpath):
    seen = set()
    out = []
    for line in open(path, encoding="utf8"):
        s = line.rstrip("\n")
        if not s or s in seen:
            continue
        if NOISE.match(s) and not CJK.search(s):
            continue
        if keep(s):
            seen.add(s)
            out.append(s)
    with open(outpath, "w", encoding="utf8") as fh:
        fh.write("\n".join(out) + "\n")
    print(outpath, len(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
