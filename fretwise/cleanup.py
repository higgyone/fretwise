"""Removing notes that cannot be part of the line around them.

Transcription occasionally reports a note far above everything near it --
a harmonic mistaken for a fundamental. These are audible as a spike when the
transcription is played back.

The test applied here is deliberately narrow: a note is dropped only when it
stands more than an octave and a half above what is being played around it.
Notes exactly an octave above their neighbours are left alone, because a
guitar genuinely doubles at the octave -- an open D chord holds D3 and D4 at
once -- and no measurement so far separates a real doubling from a spurious
one.
"""

from __future__ import annotations

from itertools import groupby

import numpy as np

# How far above the local line a note must sit to be considered impossible.
# An octave (12) is ordinary in a chord; 18 is well past any doubling.
MAX_SEMITONES_ABOVE = 18
# Seconds either side to judge "what is being played around it".
CONTEXT_WINDOW = 2.0
# Below this many neighbours there is no line to compare against.
MIN_NEIGHBOURS = 3


def local_median_midi(
    notes, index: int, *, window: float = CONTEXT_WINDOW
) -> float | None:
    """Median pitch of the notes surrounding ``notes[index]`` in time."""
    times = np.array([n.time for n in notes])
    midis = np.array([n.midi for n in notes])
    near = np.abs(times - notes[index].time) < window
    near[index] = False
    if near.sum() < MIN_NEIGHBOURS:
        return None
    return float(np.median(midis[near]))


def drop_stray_notes(
    notes,
    *,
    window: float = CONTEXT_WINDOW,
    max_semitones_above: int = MAX_SEMITONES_ABOVE,
) -> list:
    """Drop notes sitting implausibly far above the line around them.

    Only the upper side is tested. A note far *below* its neighbours would
    already have been excluded by the instrument's pitch range.
    """
    if not notes:
        return []

    kept = []
    for index, note in enumerate(notes):
        local = local_median_midi(notes, index, window=window)
        if local is None or note.midi - local < max_semitones_above:
            kept.append(note)
    return kept


# Transcription splits a held note into consecutive pieces at the same pitch.
# Measured on a strummed guitar part, the gaps between same-pitch neighbours
# are sharply bimodal: 238 pairs sit within 0.05s of each other (or overlap),
# only 4 fall between 0.05 and 0.10, and genuine re-strums resume beyond 0.1s.
# The threshold sits in that valley, so held notes join and repeated picking
# stays separate.
MAX_HELD_GAP = 0.08


def merge_held_notes(notes, *, max_gap: float = MAX_HELD_GAP) -> list:
    """Join consecutive notes at the same pitch that are really one held note.

    Only pieces that abut are merged. A gap wider than ``max_gap`` is treated
    as the string being played again, and left as two notes.
    """
    if not notes:
        return []

    merged = []
    ordered = sorted(notes, key=lambda n: (n.midi, n.time))
    for _midi, group in groupby(ordered, key=lambda n: n.midi):
        run = None
        for note in group:
            if run is not None and note.time - (run.time + run.duration) <= max_gap:
                end = max(run.time + run.duration, note.time + note.duration)
                run.duration = end - run.time
                # Keep the strongest reading: the attack represents the note
                # better than its decaying tail.
                run.confidence = max(run.confidence, note.confidence)
                continue
            run = replace_note(note)
            merged.append(run)

    merged.sort(key=lambda n: (n.time, n.midi))
    return merged


def replace_note(note):
    """A shallow copy, so merging does not mutate the caller's notes."""
    from dataclasses import replace

    return replace(note)
