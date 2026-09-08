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

# Two same-pitch pieces that abut are not always one note. On a strummed part
# the string keeps ringing while it is struck again, so transcription reports
# continuous pitch and the boundary between its pieces is the strum. Joining
# those destroys the rhythm: on one clip it welded a passage into a single
# 19 second note and emptied three seconds of playing off the fretboard.
#
# The audio settles it. A cut with no attack under it is one sound divided;
# a cut where the signal jumps is the string being played again. The measured
# rises at real strums ran 1.15 to 3.70 against about 1.0 at plain splits, so
# the line is drawn at half again as loud as the moment before.
ATTACK_RATIO = 1.5
ATTACK_WINDOW = 0.05


def attack_ratio(audio, sr: int, at: float, *, window: float = ATTACK_WINDOW) -> float:
    """How much louder the recording is just after ``at`` than just before.

    Around 1 means the sound simply continued; well above means it was struck.
    """
    frames = int(window * sr)
    index = int(at * sr)
    before = audio[max(0, index - frames) : index]
    after = audio[index : index + frames]
    if not len(before) or not len(after):
        return 1.0

    quieter = float(np.sqrt(np.mean(np.square(before))))
    louder = float(np.sqrt(np.mean(np.square(after))))
    return louder / quieter if quieter > 1e-9 else 1.0


def merge_held_notes(
    notes,
    *,
    max_gap: float = MAX_HELD_GAP,
    audio=None,
    sr: int | None = None,
    attack: float = ATTACK_RATIO,
) -> list:
    """Join consecutive notes at the same pitch that are really one held note.

    Only pieces that abut are merged. A gap wider than ``max_gap`` is the
    string being played again. So is a boundary the recording attacks at, when
    ``audio`` is given -- without it only the gap is considered, which on a
    strummed part joins repeated strikes of a ringing string into one note.
    """
    if not notes:
        return []

    def struck(at: float) -> bool:
        if audio is None or sr is None:
            return False
        return attack_ratio(audio, sr, at) >= attack

    merged = []
    ordered = sorted(notes, key=lambda n: (n.midi, n.time))
    for _midi, group in groupby(ordered, key=lambda n: n.midi):
        run = None
        for note in group:
            if (
                run is not None
                and note.time - (run.time + run.duration) <= max_gap
                and not struck(note.time)
            ):
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
