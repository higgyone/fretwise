"""Finding a repeating figure and averaging the transcription over it.

Where a bar repeats, its notes repeat with it but the transcription's mistakes
do not: a wrong note comes from one particular moment of audio and lands in
one repetition only. Folding every repetition onto one and keeping what most
of them agree on therefore strips errors that no single pass could identify.

The period is found by autocorrelating the note onsets rather than by beat
tracking, so it needs no tempo and no assumption about the meter -- what comes
out is whatever length actually repeats.

This only means anything where the playing really does repeat. Over a whole
song, sections with different figures fold on top of each other and agreement
collapses; over one section of this clip it reaches every repetition. The
agreement is reported for exactly that reason: it says whether to believe the
result.

The strongest lag is taken as the period. A two bar figure also repeats
weakly at one bar, and folding on the half loses whatever only happens in
the second, so where the reported agreement looks low it is worth trying a
multiple of the period by hand.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

# Periods shorter than this are fragments of a bar rather than a figure.
MIN_PERIOD = 1.0
MAX_PERIOD = 12.0
# Resolution of the search, and of the folded grid.
SEARCH_STEP = 0.05
DIVISIONS = 16
# How many repetitions must contain a note before it is believed.
MIN_SHARE = 0.4
# Averaging needs something to average. Agreement across two passes says
# nothing -- any coincidence is unanimous.
MIN_REPETITIONS = 4


def onset_signal(notes, step: float = SEARCH_STEP) -> np.ndarray:
    """Note starts per pitch class on a regular time grid, for autocorrelating.

    Pitch is kept rather than summed away. Onsets alone cannot tell a figure
    repeating every two seconds from notes falling every half second: only
    which pitch returns says how long the figure is.
    """
    if not notes:
        return np.zeros((12, 1))
    span = max(n.time for n in notes)
    signal = np.zeros((12, int(span / step) + 1))
    for note in notes:
        signal[note.midi % 12, int(note.time / step)] += 1
    return signal


def find_period(
    notes,
    *,
    min_period: float = MIN_PERIOD,
    max_period: float = MAX_PERIOD,
    step: float = SEARCH_STEP,
) -> tuple[float, float]:
    """The length that repeats, and how strongly. Returns ``(period, strength)``.

    Strength is the autocorrelation at that lag, so 0 is no periodicity at
    all and 1 would be a perfect loop.
    """
    signal = onset_signal(notes, step)
    if signal.shape[1] < 4 or not signal.any():
        return 0.0, 0.0

    # Each pitch class is correlated with itself and the results added, so a
    # lag only scores where the same notes come back, not merely some note.
    correlation = None
    for row in signal:
        centred = row - row.mean()
        here = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
        correlation = here if correlation is None else correlation + here

    if correlation is None or correlation[0] <= 0:
        return 0.0, 0.0
    correlation = correlation / correlation[0]

    low = int(min_period / step)
    high = min(int(max_period / step), len(correlation))
    if high <= low:
        return 0.0, 0.0

    best = int(np.argmax(correlation[low:high])) + low
    return best * step, float(correlation[best])


def fold(notes, period: float, phase: float, *, divisions: int = DIVISIONS) -> Counter:
    """Count how often each pitch falls in each slot of the repeating figure."""
    counts: Counter = Counter()
    if period <= 0:
        return counts
    for note in notes:
        position = ((note.time - phase) % period) / period
        counts[(int(round(position * divisions)) % divisions, note.midi)] += 1
    return counts


def sharpness(notes, period: float, phase: float, *, divisions: int = DIVISIONS) -> float:
    """How concentrated the folded notes are.

    A real loop stacks its notes into the same few slots, so the counts are
    tall; noise spreads evenly and gives about 1.
    """
    counts = np.array(list(fold(notes, period, phase, divisions=divisions).values()), float)
    total = counts.sum()
    return float((counts**2).sum() / total) if total else 0.0


def best_phase(notes, period: float, *, divisions: int = DIVISIONS, step: float = SEARCH_STEP) -> float:
    """Where the figure begins, chosen to stack the repetitions most tightly."""
    if period <= 0:
        return 0.0
    candidates = np.arange(0, period, step)
    return float(max(candidates, key=lambda p: sharpness(notes, period, p, divisions=divisions)))


def repetitions(notes, period: float) -> int:
    """How many times the figure goes round within the notes given."""
    if period <= 0 or not notes:
        return 0
    span = max(n.time for n in notes) - min(n.time for n in notes)
    return max(int(round(span / period)), 1)


def consensus(
    notes,
    *,
    period: float | None = None,
    phase: float | None = None,
    divisions: int = DIVISIONS,
    min_share: float = MIN_SHARE,
) -> tuple[list, dict]:
    """The notes most repetitions agree on, as one pass of the figure.

    Returns the notes and a summary: the period, phase, how many repetitions
    were folded, and how strongly the figure repeats at all.
    """
    if not notes:
        return [], {"period": 0.0, "phase": 0.0, "repetitions": 0, "strength": 0.0}

    strength = 0.0
    if period is None:
        period, strength = find_period(notes)
    if period <= 0:
        return [], {"period": 0.0, "phase": 0.0, "repetitions": 0, "strength": 0.0}
    if phase is None:
        phase = best_phase(notes, period, divisions=divisions)

    times = defaultdict(list)
    durations = defaultdict(list)
    for note in notes:
        position = ((note.time - phase) % period) / period
        slot = int(round(position * divisions)) % divisions
        times[(slot, note.midi)].append(note)
        durations[(slot, note.midi)].append(note.duration)

    total = repetitions(notes, period)
    if total < MIN_REPETITIONS:
        return [], {
            "period": round(period, 3), "phase": round(phase, 3),
            "repetitions": total, "strength": round(strength, 3),
            "divisions": divisions,
        }
    slot_length = period / divisions

    # How often each pitch appears at all, so a slot can be compared with the
    # scatter chance alone would leave there. A pitch played 40 times across 16
    # slots averages 2.5 per slot, so 6 landing together means nothing; a pitch
    # played once per repetition expects well under one, so 12 together means a
    # great deal. The margin is three standard deviations of that scatter,
    # treating the arrivals as Poisson.
    heard_at_all: Counter = Counter(note.midi for note in notes)

    kept = []
    for (slot, midi), heard in sorted(times.items()):
        share = len(heard) / total
        by_chance = heard_at_all[midi] / divisions
        if share < min_share or len(heard) < by_chance + 3 * np.sqrt(by_chance):
            continue
        # Keep a real note as the template so nothing is invented, but place
        # it on the grid and let its confidence carry the agreement.
        template = max(heard, key=lambda n: n.confidence)
        kept.append(
            replace(
                template,
                time=round(slot * slot_length, 4),
                duration=float(np.median(durations[(slot, template.midi)])),
                confidence=min(share, 1.0),
                options=None,
                chosen=None,
            )
        )

    kept.sort(key=lambda n: (n.time, n.midi))
    return kept, {
        "period": round(period, 3),
        "phase": round(phase, 3),
        "repetitions": total,
        "strength": round(strength, 3),
        "divisions": divisions,
    }
