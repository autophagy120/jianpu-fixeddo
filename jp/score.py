"""Text-score assembly: notes + (optional) lyric text -> phrases -> rendered lines.

Phrase boundaries come from the lyric punctuation (preferred - that is where a singer
breathes), falling back to geometric punctuation in the lyric band, then to barlines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import lyrics as LY
from . import music as M
from . import phrases as PH
from .core import Page, System

Span = Tuple[int, int]


@dataclass
class ScoreLine:
    tokens: List[str] = field(default_factory=list)
    chars: List[str] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)
    source: str = ""
    notes: int = 0


def octave_bias(page: Page, key_pc: int, ref_pc: int) -> int:
    """Recentre the written register: the most common octave prints without brackets.

    A bB score played on a C instrument sits a whole register above C4, which would
    otherwise put 【】 on every single note.  The header records the bias so nothing
    is lost - it is a display convention, not a pitch change.
    """
    rel = []
    for s in page.systems:
        for n in s.notes:
            if n.digit == "0":
                continue
            p = M.note_to_abs(int(n.digit), n.acc, n.octave, key_pc)
            rel.append((12 * p.octave + p.pc - (12 * 4 + ref_pc)) // 12)
    if not rel:
        return 0
    hist: Dict[int, int] = {}
    for r in rel:
        hist[r] = hist.get(r, 0) + 1
    return max(hist.items(), key=lambda kv: kv[1])[0]


def _pitch_tokens(notes, key_pc: int, ref_pc: int, harm, topts, bias: int) -> Tuple[List[str], List[int]]:
    idx_real = [i for i, n in enumerate(notes) if n.digit != "0"]
    toks = ["0"] * len(notes)
    if idx_real:
        pitches = [M.note_to_abs(int(notes[i].digit), notes[i].acc, notes[i].octave, key_pc) for i in idx_real]
        rendered = M.render_fixed_do(pitches, ref_pc, harm, topts.style,
                                     harmonica=topts.harmonica, annotate=topts.annotate,
                                     octave_bias=bias)
        for t, i in zip(rendered, idx_real):
            toks[i] = t
    return toks, idx_real


def split_by_lyric_punct(text: str, note_index: Sequence[int], max_per_char: int,
                         anchors: Optional[Sequence[int]] = None):
    """Align the line's characters to the *sounding* notes, then cut at punctuation.

    `note_index` maps sounding-note ordinal -> position in the token list (rests are
    skipped).  Returns [(a, b, chars, spans)] where a/b are token positions and spans
    are already shifted to be relative to a.
    """
    items: List[Tuple[str, bool]] = [(c, LY.is_punct(c)) for c in text]
    sung = [c for c, p in items if not p]
    if not sung or not note_index:
        return []
    if len(sung) > len(note_index):
        sung = sung[:len(note_index)]
    spans = LY.align_chars(len(note_index), sung, max_per_char=max_per_char, anchors=anchors)
    # cut after a character followed by punctuation
    cuts: List[int] = []
    k = 0
    for pos, (c, is_p) in enumerate(items):
        if is_p:
            continue
        nxt = items[pos + 1][0] if pos + 1 < len(items) else ""
        if nxt and LY.is_punct(nxt):
            cuts.append(k)
        k += 1
    bounds = [0]
    for c in cuts:
        if c + 1 < len(sung) and c + 1 > bounds[-1]:
            bounds.append(c + 1)
    bounds.append(len(sung))
    out = []
    for i in range(len(bounds) - 1):
        c0, c1 = bounds[i], bounds[i + 1]
        if c1 <= c0:
            continue
        s0 = spans[c0][0]
        s1 = spans[c1 - 1][1]
        a = note_index[s0]
        b = note_index[min(s1, len(note_index) - 1)] + 1
        local = []
        for x, y in spans[c0:c1]:
            lo = max(0, min(note_index[min(x, len(note_index) - 1)] - a, b - a - 1))
            hi = min(b - a, note_index[min(y - 1, len(note_index) - 1)] - a + 1)
            local.append((lo, max(lo + 1, hi)))
        out.append((a, b, [ch for _i, ch in zip(range(c0, c1), sung[c0:c1])], local))
    if out:
        # pickup notes before the first syllable belong to the first phrase
        a0, b0, c0_, sp0 = out[0]
        out[0] = (0, b0, c0_, local_shift(sp0, a0))
    return out


def local_shift(spans: Sequence[Span], by: int) -> List[Span]:
    return [(max(0, a - by), max(1, b - by)) for a, b in spans]


def build_page(
    page: Page,
    key_pc: int,
    ref_pc: int,
    harm: M.HarmonicaOptions,
    topts,
    lyric_per_system: Optional[Sequence[str]] = None,
    bars_per_phrase: int = 4,
    max_per_char: int = 4,
    use_image_punct: bool = True,
) -> Tuple[List[Tuple[int, List[ScoreLine]]], int]:
    bias = octave_bias(page, key_pc, ref_pc)
    out: List[Tuple[int, List[ScoreLine]]] = []
    for si, sysn in enumerate(page.systems):
        lines: List[ScoreLine] = []
        if not sysn.notes:
            out.append((si, lines))
            continue
        toks, idx_real = _pitch_tokens(sysn.notes, key_pc, ref_pc, harm, topts, bias)
        text = lyric_per_system[si] if lyric_per_system and si < len(lyric_per_system) else ""
        if text:
            anchors: List[int] = []
            if use_image_punct:
                for x, _h in PH.punctuation_marks(page, sysn, page.scale):
                    k = -1
                    for j, nl in enumerate(idx_real):
                        if sysn.notes[nl].x <= x:
                            k = j
                    if k >= 0:
                        anchors.append(k)
            chunks = split_by_lyric_punct(text, idx_real, max_per_char, anchors)
            if chunks:
                for a, b, chars, spans in chunks:
                    lines.append(ScoreLine(tokens=toks[a:b], chars=chars, spans=spans,
                                           source="歌词标点", notes=b - a))
                tail = chunks[-1][1]
                if tail < len(toks):
                    lines.append(ScoreLine(tokens=toks[tail:], source="行尾无词", notes=len(toks) - tail))
                out.append((si, lines))
                continue
        spans, how = PH.split_notes(page, sysn, page.scale, bars_per_phrase)
        for a, b in spans:
            lines.append(ScoreLine(tokens=toks[a:b], chars=[], spans=[], source=how, notes=b - a))
        out.append((si, lines))
    return out, bias


def render_page(page_lines, mode: str = "align", continuation: str = "～") -> List[str]:
    out: List[str] = []
    for si, lines in page_lines:
        if not lines:
            out.append(f"# [第{si + 1}行] 无音符，需人工确认")
            continue
        for ln in lines:
            if ln.chars:
                out += LY.render(mode, ln.tokens, ln.chars, ln.spans, continuation)
            else:
                out.append(" ".join(ln.tokens))
        out.append("")
    return out
