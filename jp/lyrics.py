"""Lyric alignment and the four ways to render 音符 + 歌词 in a plain-text score.

One Chinese character can own several notes (拖腔 / melisma).  That is the only hard
part, and it drives the layout choice:

    align  1 2 3 5 6        <- each character sits under the first note of its span,
           春 ～ ～ 风 月      the rest get a continuation mark
    group  [1 2 3]春 [5 6]风 [1]月
    inline 1春 2 3 5风 6月
    tsv    1\\t春 / 2\\t (empty) ...

`align` needs display-width arithmetic: 汉字是双宽, `#1`/`（1）`/`【1】` 宽度都不一样。
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

PUNCT = "，。、；：！？…—～·「」『』（）《》〈〉【】,.;:!?()[]{}"

Span = Tuple[int, int]        # [a, b) note indices owned by one character


def dwidth(s: str) -> int:
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def is_punct(ch: str) -> bool:
    return ch in PUNCT


# ------------------------------------------------------------------ alignment
def align_chars(
    n_notes: int,
    chars: Sequence[str],
    max_per_char: int = 4,
    anchors: Optional[Sequence[int]] = None,
) -> List[Span]:
    """Distribute `chars` over `n_notes` notes: contiguous, non-empty, order kept.

    Cost: deviation from an even split (so 拖腔 spreads instead of piling up),
    plus a small bonus when a character starts on an anchor note (标点/小节线后的音).
    """
    m, n = len(chars), n_notes
    if m == 0 or n == 0:
        return []
    if m > n:
        # more syllables than notes: the pairing is wrong somewhere; give every note
        # one syllable and let the caller truncate (reported as a warning upstream)
        return [(i, i + 1) for i in range(n)]
    even = n / m
    INF = 1e18
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    back = [[0] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    anchor_set = set(anchors or [])
    for i in range(1, m + 1):
        for j in range(i, n + 1):
            for k in range(1, min(max_per_char, j - i + 1) + 1):
                prev = dp[i - 1][j - k]
                if prev >= INF:
                    continue
                c = prev + (k - even) ** 2
                if anchor_set and (j - k) in anchor_set:
                    c -= 0.35
                if c < dp[i][j]:
                    dp[i][j] = c
                    back[i][j] = k
    if dp[m][n] >= INF:                     # max_per_char too small: relax
        return align_chars(n_notes, chars, max_per_char=max_per_char + 2, anchors=anchors)
    spans = []
    j, i = n, m
    while i > 0:
        k = back[i][j]
        spans.append((j - k, j))
        j -= k
        i -= 1
    return list(reversed(spans))


# ------------------------------------------------------------------ rendering
def _token_cols(tokens: Sequence[str], gap: int = 1) -> Tuple[List[int], List[int], int]:
    starts, widths = [], []
    col = 0
    for t in tokens:
        w = dwidth(t)
        starts.append(col)
        widths.append(w)
        col += w + gap
    return starts, widths, max(0, col - gap)


def render_align(
    tokens: Sequence[str],
    chars: Sequence[str],
    spans: Sequence[Span],
    continuation: str = "～",
    gap: int = 1,
    indent: str = "",
) -> List[str]:
    """Two aligned rows; continuation marks sit under the extra notes of a 拖腔."""
    starts, widths, total = _token_cols(tokens, gap)
    if not tokens:
        return []
    row = [" "] * (total + 2)
    cont = [" "] * (total + 2)
    last_end = 0
    for ch, (a, b) in zip(chars, spans):
        a = max(0, min(a, len(tokens) - 1))
        b = max(a + 1, min(b, len(tokens)))
        cw = dwidth(ch)
        left = starts[a]
        right = starts[b - 1] + widths[b - 1]
        pos = int(round((left + right) / 2.0 - cw / 2.0))
        pos = max(pos, last_end)
        if pos + cw > len(row):
            row += [" "] * (pos + cw - len(row) + 2)
            cont += [" "] * (pos + cw - len(cont) + 2)
        for k, c in enumerate(ch):
            row[pos + k] = c
        last_end = pos + cw + 1
        if b - a > 1:
            for t in range(a + 1, b):
                cm = dwidth(continuation)
                cpos = starts[t] + max(0, (widths[t] - cm) // 2)
                for k, c in enumerate(continuation):
                    if cpos + k < len(cont):
                        cont[cpos + k] = c
    top = indent + " ".join(tokens)
    bottom = indent + "".join(row).rstrip()
    contline = indent + "".join(cont).rstrip()
    out = [top, bottom]
    if contline.strip():
        out.append(contline)
    return out


def render_group(tokens: Sequence[str], chars: Sequence[str], spans: Sequence[Span]) -> List[str]:
    parts = []
    for ch, (a, b) in zip(chars, spans):
        parts.append("[" + " ".join(tokens[a:b]) + "]" + ch)
    covered = spans[-1][1] if spans else 0
    if covered < len(tokens):
        parts.append(" ".join(tokens[covered:]))
    return [" ".join(parts)]


def render_inline(tokens: Sequence[str], chars: Sequence[str], spans: Sequence[Span]) -> List[str]:
    owned: List[str] = [""] * len(tokens)
    for ch, (a, b) in zip(chars, spans):
        if a < len(owned):
            owned[a] = ch
    return [" ".join(t + o for t, o in zip(tokens, owned))]


def render_tsv(tokens: Sequence[str], chars: Sequence[str], spans: Sequence[Span]) -> List[str]:
    owner: List[str] = [""] * len(tokens)
    for ch, (a, b) in zip(chars, spans):
        if a < len(owner):
            owner[a] = ch
    return ["\t".join(f"{t}\t{o}" for t, o in zip(tokens, owner))]


def render(
    mode: str,
    tokens: Sequence[str],
    chars: Sequence[str],
    spans: Sequence[Span],
    continuation: str = "～",
) -> List[str]:
    if not chars:
        return [" ".join(tokens)]
    if mode == "group":
        return render_group(tokens, chars, spans)
    if mode == "inline":
        return render_inline(tokens, chars, spans)
    if mode == "tsv":
        return render_tsv(tokens, chars, spans)
    return render_align(tokens, chars, spans, continuation)


# ------------------------------------------------------------------ text input
def split_phrases(text: str) -> List[str]:
    """'春风花雨，月落乌啼。' -> ['春风花雨，', '月落乌啼。']  (punctuation stays)."""
    text = re.sub(r"\s+", "", text)
    out, cur = [], ""
    for ch in text:
        cur += ch
        if is_punct(ch):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return [p for p in out if p.strip(PUNCT)]


def lyric_chars(phrase: str, keep_punct: bool = False) -> List[str]:
    return [c for c in phrase if keep_punct or not is_punct(c)]


def load_file(path: str) -> List[str]:
    with open(path, encoding="utf8") as fh:
        raw = fh.read()
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("#")]
    if len(lines) > 1:
        return lines
    return split_phrases(raw)


def fit_to_systems(phrases: Sequence[str], n_systems: int) -> List[str]:
    """One phrase per music line; pad/merge when the counts disagree."""
    if n_systems <= 0:
        return []
    if not phrases:
        return [""] * n_systems
    if len(phrases) == n_systems:
        return list(phrases)
    if len(phrases) > n_systems:
        # merge the tail so nothing is lost
        out = list(phrases[:n_systems - 1])
        out.append("".join(phrases[n_systems - 1:]))
        return out
    out = list(phrases)
    while len(out) < n_systems:
        out.append("")
    return out
