"""Music theory: keys, movable-do <-> fixed-do, enharmonic spelling, harmonica fingering.

A note in this pipeline is `(degree, accidental, octave)` where degree is the jianpu
number 1-7, accidental is one of {'', '#', 'b'} and octave is the dot count (>=0 up).

Two different renderings come out of one absolute pitch:

* **movable do** (the transposed image): degree relative to the *target key*.
* **fixed do** (the text score): degree relative to C (or C#), i.e. 1 is always a
  specific key on the keyboard.  For a chromatic harmonica this is the useful one:
  a naked digit maps to hole+breath, and the accidental tells you to press the slide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# semitone distance from the tonic for the seven scale degrees of a major key
DEG_SEMI = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
SEMI_DEG = {v: k for k, v in DEG_SEMI.items()}

PC_OF = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PC_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ---- chromatic harmonica (solo tuning, key of C, 12 hole) -------------------
# blow: 1(C) 3(E) 5(G) 7? -> standard layout: blow C E G C, draw D F A B
BLOW_DEGREES = {1, 3, 5}
FLAT_SYMBOL = ""


def breath_of_degree(deg: int) -> str:
    return "吹" if deg in BLOW_DEGREES else "吸"


def parse_key(text: str) -> Optional[int]:
    """'bA' / '#G' / 'C' / 'bE' / 'F#' -> pitch class of the tonic."""
    if not text:
        return None
    t = text.strip().replace("♭", "b").replace("♯", "#").replace("＃", "#").replace("＝", "=")
    t = t.upper()
    acc = 0
    if t.startswith("B") and len(t) > 1 and t[1] in "ABCDEFG":
        acc, t = -1, t[1:]
    elif t.startswith("#"):
        acc, t = 1, t[1:]
    elif t.startswith("B") and len(t) == 1:
        acc, t = 0, "B"
    if not t or t[0] not in PC_OF:
        return None
    pc = PC_OF[t[0]]
    if len(t) > 1 and t[1] in "#b":
        acc = 1 if t[1] == "#" else -1
    return (pc + acc) % 12


def key_name(pc: int, prefer_flat: bool = False) -> str:
    flats = {1: "bD", 3: "bE", 6: "bG", 8: "bA", 10: "bB"}
    if prefer_flat and pc in flats:
        return flats[pc]
    names = {0: "C", 1: "#C", 2: "D", 3: "#D", 4: "E", 5: "F", 6: "#F", 7: "G", 8: "#G", 9: "A", 10: "#A", 11: "B"}
    return names[pc]


@dataclass
class Pitch:
    """Absolute pitch of one jianpu note."""
    pc: int                  # 0-11
    octave: int              # 4 = the octave containing C4..B4
    deg: int                 # source degree 1-7 (0 = rest)
    acc: str = ""

    @property
    def midi(self) -> int:
        return 12 * (self.octave + 1) + self.pc


def note_to_abs(deg: int, acc: str, octave: int, tonic_pc: int) -> Pitch:
    """movable-do note -> absolute pitch.  `octave` is the dot count (+ up)."""
    a = 1 if acc == "#" else (-1 if acc == "b" else 0)
    semi = DEG_SEMI.get(deg, 0) + a
    base_oct = 4 + octave
    total = 12 * base_oct + tonic_pc + semi
    return Pitch(pc=total % 12, octave=total // 12, deg=deg, acc=acc)


def abs_to_movable(p: Pitch, tonic_pc: int, prefer_flat: bool = False) -> Tuple[int, str, int]:
    """absolute pitch -> (degree, accidental, octave dots) in the given key.

    Ties (b7 vs #6) are broken towards sharps by default - that is the convention
    jianpu transposition charts use (chengdu D->C writes #4, not b5).
    """
    total = 12 * p.octave + p.pc
    ref = 12 * 4 + tonic_pc            # the tonic in the "middle" octave
    rel = total - ref
    octave = int(rel // 12)
    within = rel - 12 * octave          # 0..11
    best = None
    for deg, semi in DEG_SEMI.items():
        for acc, delta in (("", 0), ("#", 1), ("b", -1)):
            if semi + delta == within:
                tie = 0 if delta == 0 else ((2 if prefer_flat else 1) if delta == 1 else (1 if prefer_flat else 2))
                cost = 10 * abs(delta) + tie
                if best is None or cost < best[0]:
                    best = (cost, deg, acc)
    if best is None:                    # double accidental: spell as # of the lower degree
        deg = max((d for d, s in DEG_SEMI.items() if s <= within), key=lambda d: DEG_SEMI[d])
        best = (99, deg, "#")
    return best[1], best[2], octave


# ------------------------------------------------------------------ fixed do
def degree_candidates(semi_from_ref: int) -> List[Tuple[int, str]]:
    """All spellings (degree, accidental) in the fixed-do system whose pitch equals
    `semi_from_ref` semitones above the reference (C or C#)."""
    out = []
    s = semi_from_ref % 12
    for deg, semi in DEG_SEMI.items():
        for acc, delta in (("", 0), ("#", 1), ("b", -1)):
            if (semi + delta) % 12 == s:
                out.append((deg, acc))
    # prefer the plain degree first, then sharps, then flats
    order = {"": 0, "#": 1, "b": 2}
    out.sort(key=lambda t: (order[t[1]], t[0]))
    return out


def fingering(deg: int, acc: str) -> Tuple[str, int]:
    """Chromatic harmonica (C solo tuning): (吹/吸, slide 0/1).

    A flat on degree d is played as the sharp of the degree below, so
    b1 == #7 (draw + slide) and b4 == #3 (blow + slide).
    """
    if acc == "#":
        return breath_of_degree(deg), 1
    if acc == "b":
        lower = deg - 1 if deg > 1 else 7
        return breath_of_degree(lower), 1
    return breath_of_degree(deg), 0


@dataclass
class HarmonicaOptions:
    breath_penalty: float = 1.0      # 换气方向 (吹<->吸)
    slide_penalty: float = 0.6       # 推键切换
    accidental_penalty: float = 0.25  # 多一个升降号的可读性代价
    prefer_flat: bool = False        # 等音并列时优先降号
    allow_degree_swap: bool = True   # 允许 #3<->4 这类跨音级等音


def choose_spelling(
    semi: int,
    opts: HarmonicaOptions,
) -> List[Tuple[int, str]]:
    cands = degree_candidates(semi)
    if not opts.allow_degree_swap:
        plain = [c for c in cands if c[1] == ""]
        if plain:
            return plain
        cands = [c for c in cands if c[1] == "#"]
    return cands


def solve_harmonica(
    semis: Sequence[int],
    opts: HarmonicaOptions = HarmonicaOptions(),
) -> List[Tuple[int, str]]:
    """Pick one spelling per pitch so the breath/slide pattern is smooth.

    Dynamic programming over the (usually 2, at most 3) spellings of each pitch:
    cost = breath change + slide change + accidental count.
    """
    if not semis:
        return []
    cand = [choose_spelling(s, opts) for s in semis]
    n = len(semis)
    INF = 1e9
    cost = [[0.0] * len(cand[i]) for i in range(n)]
    back = [[-1] * len(cand[i]) for i in range(n)]
    for j, (d, a) in enumerate(cand[0]):
        cost[0][j] = opts.accidental_penalty * (0 if a == "" else 1)
        if a == "b" and not opts.prefer_flat:
            cost[0][j] += 0.05
        if a == "#" and opts.prefer_flat:
            cost[0][j] += 0.05
    for i in range(1, n):
        for j, (d, a) in enumerate(cand[i]):
            br, sl = fingering(d, a)
            base = opts.accidental_penalty * (0 if a == "" else 1)
            if a == "b" and not opts.prefer_flat:
                base += 0.05
            if a == "#" and opts.prefer_flat:
                base += 0.05
            best, bestk = INF, -1
            for k, (pd, pa) in enumerate(cand[i - 1]):
                pbr, psl = fingering(pd, pa)
                c = cost[i - 1][k] + base
                if pbr != br:
                    c += opts.breath_penalty
                if psl != sl:
                    c += opts.slide_penalty
                if c < best:
                    best, bestk = c, k
            cost[i][j], back[i][j] = best, bestk
    j = min(range(len(cand[-1])), key=lambda k: cost[-1][k])
    out = [cand[-1][j]]
    for i in range(n - 1, 0, -1):
        j = back[i][j]
        out.append(cand[i - 1][j])
    return list(reversed(out))


def accidental_count(tokens: Iterable[Tuple[int, str]]) -> int:
    return sum(1 for _, a in tokens if a)


def pick_reference(pitches: Sequence[Pitch], threshold: int = 3) -> Tuple[int, str]:
    """Choose the fixed-do reference: C, or C# once C needs too many accidentals.

    Matches the user's rule: 转成固定调后若升号超过 2~3 个，就改用 C# 计谱。
    Returns (reference_pc, reason).
    """
    def cost(ref: int) -> int:
        n = 0
        for p in pitches:
            semi = (p.pc - ref) % 12
            best = min(degree_candidates(semi), key=lambda t: 0 if t[1] == "" else 1)
            n += 0 if best[1] == "" else 1
        return n

    c_cost = cost(0)
    cs_cost = cost(1)
    if c_cost > threshold and cs_cost < c_cost:
        return 1, f"C 记谱需要 {c_cost} 个升降号 > {threshold}，改用 C# 记谱（{cs_cost} 个）"
    return 0, f"C 记谱需要 {c_cost} 个升降号（C# 记谱需 {cs_cost} 个）"


def format_note(deg: int, acc: str, octave: int, style: str = "cn") -> str:
    """Render one note.  低八度：（1）  高八度：【1】"""
    if style == "cn":
        lp, rp, lh, rh = "（", "）", "【", "】"
    else:
        lp, rp, lh, rh = "(", ")", "[", "]"
    core = f"{acc}{deg}"
    if octave == 0:
        return core
    if octave > 0:
        return lh * min(octave, 2) + core + rh * min(octave, 2)
    return lp * min(-octave, 2) + core + rp * min(-octave, 2)


def render_fixed_do(
    pitches: Sequence[Pitch],
    ref_pc: int,
    opts: HarmonicaOptions = HarmonicaOptions(),
    style: str = "cn",
    harmonica: bool = True,
    annotate: bool = False,
    octave_bias: int = 0,
) -> List[str]:
    semis = [(p.pc - ref_pc) % 12 for p in pitches]
    spells = solve_harmonica(semis, opts) if harmonica else [
        min(degree_candidates(s), key=lambda t: 0 if t[1] == "" else 1) for s in semis
    ]
    out = []
    for p, (deg, acc) in zip(pitches, spells):
        # octave relative to the reference: the reference sits in octave 4
        ref_abs = 12 * 4 + ref_pc
        rel = (12 * p.octave + p.pc) - ref_abs
        octave = rel // 12 - octave_bias
        txt = format_note(deg, acc, octave, style)
        if annotate:
            br, sl = fingering(deg, acc)
            txt += f"({br}{'推' if sl else ''})"
        out.append(txt)
    return out


def transpose_movable(deg: int, acc: str, octave: int, key_from_pc: int,
                      key_to_pc: int, octave_mode: str = "keep") -> Tuple[int, str, int]:
    """首调记谱重写：只改数字（+临时升降号），**书写八度沿用原谱**。

    与 easytr 的做法一致：转调 = 把级数按两个主音的音程推过去，
    只有级数越过主音时才进位/退位八度点，不会因为"绝对音高落在哪个八度"而满谱添点。
    例：原调 F(5) → 目标 C(0)，s=5；1→4、7→3^、5→1、3→6。
    """
    delta = 1 if acc == "#" else (-1 if acc == "b" else 0)
    base = DEG_SEMI.get(deg, 0) + delta
    s = (key_from_pc - key_to_pc) % 12
    total = base + s
    wrap = total // 12                 # 0 或 1（base 可为 -1，所以也可能是 -1）
    within = total - 12 * wrap
    best = None
    for d, semi in DEG_SEMI.items():
        for a, dd in (("", 0), ("#", 1), ("b", -1)):
            if semi + dd == within:
                cost = abs(dd) * 10 + (0 if dd == 0 else 1)
                if best is None or cost < best[0]:
                    best = (cost, d, a)
    if best is None:
        return deg, acc, octave
    if octave_mode == "keep":
        # 与 easytr 一致：八度点原样保留，只换数字
        return best[1], best[2], octave
    if octave_mode == "wrap":
        return best[1], best[2], octave + wrap
    return best[1], best[2], octave + wrap
