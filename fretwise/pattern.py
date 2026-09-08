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
# Slots per figure. At 16 a six second figure gives slots of 0.4s, coarser
# than the playing, and several notes of one repetition fall in the same one.
DIVISIONS = 32
# How many repetitions must contain a note before it is believed.
MIN_SHARE = 0.4
# Averaging needs something to average. Agreement across two passes says
# nothing -- any coincidence is unanimous.
MIN_REPETITIONS = 4
# A share alone says too little when there are few repetitions: two of four
# already reads as 50%. What matters is how many repetitions actually contain
# an average note, so agreement and repetitions are judged together - a kept
# note must be there in about three passes, whether that is three of four or
# five of nine.
MIN_EVIDENCE = 3.0
# How far a figure must beat the agreement that shuffled timing reaches. On
# this material the real sections clear their own null by 0.15 and 0.24, while
# scattered playing ties it exactly, so a small margin separates them.
CHANCE_MARGIN = 0.05
# A section long enough to hold several repetitions of a figure. Too short and
# nothing repeats often enough to average; too long and two different figures
# fold on top of each other.
SECTION = 40.0
# Below this much agreement a section is left exactly as transcribed: whatever
# is there does not repeat, so averaging it would invent a figure.
MIN_AGREEMENT = 0.5


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


def cluster_positions(entries, tolerance: float):
    """Group notes of one pitch whose folded positions sit within ``tolerance``.

    Fixed bins split a note that arrives a little early in one repetition from
    the same note arriving a little late in another, and neither half then
    looks agreed on. Clustering follows the playing instead of a grid.
    """
    clusters: list[list] = []
    for position, turn, note in sorted(entries):
        # Measured from where the cluster began, not from its last member:
        # comparing against the last lets a chain of small steps grow a
        # cluster without limit, so scattered playing merges into one.
        if clusters and position - clusters[-1][0][0] <= tolerance:
            clusters[-1].append((position, turn, note))
        else:
            clusters.append([(position, turn, note)])
    return clusters


def consensus(
    notes,
    *,
    period: float | None = None,
    phase: float | None = None,
    divisions: int = DIVISIONS,
    min_share: float = MIN_SHARE,
    min_repetitions: int = MIN_REPETITIONS,
) -> tuple[list, dict]:
    """The notes most repetitions agree on, as one pass of the figure.

    Agreement counts how many *repetitions* contain the note, not how many
    times it was played: a pitch struck five times within one bar must not
    look like five repetitions agreeing.

    Returns the notes and a summary: the period, phase, how many repetitions
    were folded, and how strongly the figure repeats at all.
    """
    empty = {"period": 0.0, "phase": 0.0, "repetitions": 0, "strength": 0.0,
             "divisions": divisions}
    if not notes:
        return [], empty

    strength = 0.0
    if period is None:
        period, strength = find_period(notes)
    if period <= 0:
        return [], empty
    if phase is None:
        phase = best_phase(notes, period, divisions=divisions)

    total = repetitions(notes, period)
    if total < min_repetitions:
        return [], {"period": round(period, 3), "phase": round(phase, 3),
                    "repetitions": total, "strength": round(strength, 3),
                    "divisions": divisions}

    # Fold every note, remembering which turn of the figure it came from.
    by_pitch: dict[int, list] = defaultdict(list)
    for note in notes:
        elapsed = note.time - phase
        turn = int(np.floor(elapsed / period))
        by_pitch[note.midi].append(((elapsed % period) / period, turn, note))

    tolerance = 1.0 / divisions
    kept = []
    for _midi, entries in sorted(by_pitch.items()):
        for cluster in cluster_positions(entries, tolerance):
            turns = {turn for _pos, turn, _note in cluster}
            share = len(turns) / total
            if share < min_share:
                continue

            heard = [note for _pos, _turn, note in cluster]
            template = max(heard, key=lambda n: n.confidence)
            centre = float(np.median([pos for pos, _t, _n in cluster]))
            kept.append(
                replace(
                    template,
                    time=round(centre * period, 4),
                    duration=float(np.median([n.duration for n in heard])),
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


def apply_to_timeline(
    notes,
    figure: list,
    summary: dict,
    *,
    start: float | None = None,
    end: float | None = None,
) -> tuple[list, dict]:
    """Play the agreed figure at every repetition, in place of what was read.

    This is the point of averaging: a note the transcription missed in one
    repetition is restored from the others, and a note it invented once is
    dropped. It also flattens any real variation between repetitions, so it
    is worth only where the playing genuinely repeats -- which is what the
    agreement figure is for.

    Notes outside the range are left exactly as they were.
    """
    period = summary.get("period", 0.0)
    if period <= 0 or not figure:
        return list(notes), {"replaced": 0, "added": 0, "removed": 0}

    inside = [n for n in notes if _within(n, start, end)]
    outside = [n for n in notes if not _within(n, start, end)]
    if not inside:
        return list(notes), {"replaced": 0, "added": 0, "removed": 0}

    first = min(n.time for n in inside)
    last = max(n.time for n in inside)
    phase = summary.get("phase", 0.0)

    # Start at the first whole repetition at or before the earliest note.
    turns = int(np.floor((first - phase) / period))
    rebuilt = []
    while phase + turns * period <= last:
        origin = phase + turns * period
        for note in figure:
            at = origin + note.time
            if at < first - period or at > last + period:
                continue
            if not _within_time(at, start, end):
                continue
            rebuilt.append(replace(note, time=round(at, 4), options=None, chosen=None))
        turns += 1

    rebuilt.sort(key=lambda n: (n.time, n.midi))
    return sorted(outside + rebuilt, key=lambda n: (n.time, n.midi)), {
        "replaced": len(inside),
        "added": max(len(rebuilt) - len(inside), 0),
        "removed": max(len(inside) - len(rebuilt), 0),
        "rebuilt": len(rebuilt),
    }


def _within(note, start, end) -> bool:
    return _within_time(note.time, start, end)


def _within_time(at: float, start, end) -> bool:
    if start is not None and at < start:
        return False
    if end is not None and at >= end:
        return False
    return True


def best_period(
    notes,
    *,
    divisions: int = DIVISIONS,
    min_share: float = MIN_SHARE,
    max_period: float = MAX_PERIOD,
) -> float:
    """Choose between the strongest lag and its multiples by what they yield.

    Correlation cannot separate a two bar figure from the one bar inside it -
    both score well - but folding on the wrong one is obvious afterwards:
    everything that happens only in the second bar appears in half the
    repetitions, and agreement falls. So the candidates are tried and judged
    on the agreement they actually produce.
    """
    period, _strength = find_period(notes)
    if period <= 0:
        return 0.0

    best, best_score = period, -1.0
    for factor in (1, 2, 3, 4):
        candidate = period * factor
        if candidate > max_period:
            break
        figure, summary = consensus(
            notes, period=candidate, divisions=divisions, min_share=min_share
        )
        if not figure:
            continue
        agreement = sum(n.confidence for n in figure) / len(figure)
        # A longer figure fits fewer times into the same stretch, so it must
        # still be seen often enough to be worth believing.
        if agreement * summary.get("repetitions", 0) < MIN_EVIDENCE:
            continue
        # Agreement alone would favour a tiny figure that repeats trivially,
        # so the number of notes recovered counts too.
        score = agreement * len(figure)
        if score > best_score:
            best, best_score = candidate, score
    return best


def chance_agreement(
    notes,
    period: float,
    *,
    divisions: int = DIVISIONS,
    min_share: float = MIN_SHARE,
    trials: int = 5,
    seed: int = 0,
) -> float:
    """What agreement this much playing reaches when its timing is shuffled.

    Clusters are chosen after the fact, always the best-agreeing ones, so some
    agreement arises from nothing at all - the denser the playing the more.
    Rather than guess a threshold, the same measurement is run on the same
    notes with their times scattered, and the real figure has to beat it.
    """
    if period <= 0 or not notes:
        return 0.0

    rng = np.random.default_rng(seed)
    first = min(n.time for n in notes)
    last = max(n.time for n in notes)
    best = 0.0

    for _ in range(trials):
        shuffled = [
            replace(note, time=float(rng.uniform(first, last))) for note in notes
        ]
        figure, _summary = consensus(
            shuffled, period=period, divisions=divisions, min_share=min_share
        )
        if figure:
            best = max(best, sum(n.confidence for n in figure) / len(figure))
    return best


# Bars are compared on a coarser grid than notes are averaged on: transcribed
# timing jitters, and at 32 slots two performances of the same bar stop looking
# alike. Measured on this clip, 16 slots separates neighbouring bars from
# unrelated ones best.
RUN_DIVISIONS = 16
# How far above the similarity of unrelated bars two neighbours must sit.
RUN_SIGMAS = 1.0
# A run shorter than this is not worth averaging.
MIN_RUN_BARS = 3
# Bars this alike are the same bar by any standard. The threshold is capped
# here because the null breaks down when a piece is one figure throughout:
# its unrelated bars are identical too, and the bar to beat rises above 1.
ALIKE_ENOUGH = 0.9


def bar_signatures(notes, period: float, phase: float, *, divisions: int = RUN_DIVISIONS):
    """Each bar as the set of (slot, pitch) cells it contains."""
    signatures: dict[int, set] = defaultdict(set)
    for note in notes:
        index = int(np.floor((note.time - phase) / period))
        position = ((note.time - phase) % period) / period
        signatures[index].add((int(position * divisions) % divisions, note.midi))
    return {index: cells for index, cells in signatures.items() if index >= 0}


def similarity(one: set, other: set) -> float:
    """Share of cells two bars have in common, of all the cells they use."""
    if not one and not other:
        return 1.0
    return len(one & other) / max(len(one | other), 1)


def unrelated_similarity(signatures: dict, *, trials: int = 600, seed: int = 0):
    """How alike two bars look when they have nothing to do with each other.

    Bars are never identical once transcribed, so alikeness has to be judged
    against what unrelated bars of this same piece already score.
    """
    order = sorted(signatures)
    if len(order) < 4:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    scores = [
        similarity(signatures[a], signatures[b])
        for a, b in zip(rng.choice(order, trials), rng.choice(order, trials))
        if abs(int(a) - int(b)) > 1
    ]
    return (float(np.mean(scores)), float(np.std(scores))) if scores else (0.0, 0.0)


def find_runs(
    notes,
    period: float,
    phase: float,
    *,
    divisions: int = RUN_DIVISIONS,
    sigmas: float = RUN_SIGMAS,
    min_bars: int = MIN_RUN_BARS,
) -> list[list[int]]:
    """Stretches of consecutive bars that are alike, so worth averaging together.

    Each bar is compared with the one that began the run rather than with the
    one before it, so a run cannot drift into different playing a bar at a
    time.
    """
    signatures = bar_signatures(notes, period, phase, divisions=divisions)
    order = sorted(signatures)
    if len(order) < min_bars:
        return []

    mean, deviation = unrelated_similarity(signatures)
    threshold = min(mean + sigmas * deviation, ALIKE_ENOUGH)

    runs, run = [], [order[0]]
    for index in order[1:]:
        alike = similarity(signatures[run[0]], signatures[index]) >= threshold
        if alike and index - 1 == run[-1]:
            run.append(index)
        else:
            runs.append(run)
            run = [index]
    runs.append(run)
    return [r for r in runs if len(r) >= min_bars]


def average_runs(
    notes,
    *,
    period: float | None = None,
    divisions: int = DIVISIONS,
    min_share: float = MIN_SHARE,
    sigmas: float = RUN_SIGMAS,
    min_bars: int = MIN_RUN_BARS,
) -> tuple[list, list[dict]]:
    """Find runs of alike bars anywhere in the piece and average each one.

    Nothing has to be told where the sections are: a bar length is found for
    the piece, consecutive bars are compared, and only stretches that really
    do repeat are touched.
    """
    if not notes:
        return [], []

    if period is None:
        period, _strength = find_period(notes)
    if period <= 0:
        return list(notes), []
    phase = best_phase(notes, period)

    runs = find_runs(notes, period, phase, sigmas=sigmas, min_bars=min_bars)
    if not runs:
        return list(notes), []

    spans = [
        (phase + run[0] * period, phase + (run[-1] + 1) * period, len(run))
        for run in runs
    ]

    result = [
        n for n in notes if not any(a <= n.time < b for a, b, _bars in spans)
    ]
    reports = []
    for first, last, bars in spans:
        inside = [n for n in notes if first <= n.time < last]
        figure, summary = consensus(
            inside, period=period, divisions=divisions, min_share=min_share,
            min_repetitions=min_bars,
        )
        report = {"start": round(first, 2), "end": round(last, 2), "bars": bars,
                  "before": len(inside), "after": len(inside), "averaged": False,
                  "agreement": 0.0}
        if figure:
            rebuilt, _stats = apply_to_timeline(
                inside, figure, summary, start=first, end=last
            )
            result.extend(rebuilt)
            report["after"] = len(rebuilt)
            report["averaged"] = True
            report["agreement"] = round(
                sum(n.confidence for n in figure) / len(figure), 3
            )
        else:
            result.extend(inside)
        reports.append(report)

    result.sort(key=lambda n: (n.time, n.midi))
    return result, reports


def average_sections(
    notes,
    *,
    section: float = SECTION,
    divisions: int = DIVISIONS,
    min_share: float = MIN_SHARE,
    min_agreement: float = MIN_AGREEMENT,
    start: float | None = None,
    end: float | None = None,
) -> tuple[list, list[dict]]:
    """Average each stretch of the piece against its own repeating figure.

    A song does not repeat one figure throughout: verse and chorus fold on
    top of each other and agreement collapses. Each section is therefore
    given its own period, and a section that does not repeat well enough is
    left exactly as it was rather than having a figure imposed on it.

    Returns the notes and one report per section.
    """
    if not notes:
        return [], []

    first = start if start is not None else min(n.time for n in notes)
    last = end if end is not None else max(n.time for n in notes) + 0.001

    result = [n for n in notes if n.time < first or n.time >= last]
    reports = []

    edge = first
    while edge < last:
        stop = min(edge + section, last)
        inside = [n for n in notes if edge <= n.time < stop]
        report = {"start": round(edge, 2), "end": round(stop, 2),
                  "before": len(inside), "after": len(inside),
                  "period": 0.0, "agreement": 0.0, "averaged": False}

        period = best_period(inside, divisions=divisions, min_share=min_share)
        figure, summary = consensus(
            inside, period=period or None, divisions=divisions, min_share=min_share
        )
        agreement = (
            sum(n.confidence for n in figure) / len(figure) if figure else 0.0
        )
        report["period"] = summary.get("period", 0.0)
        report["agreement"] = round(agreement, 3)
        report["repetitions"] = summary.get("repetitions", 0)
        report["evidence"] = round(agreement * summary.get("repetitions", 0), 2)

        evidence = agreement * summary.get("repetitions", 0)
        chance = chance_agreement(
            inside, summary.get("period", 0.0), divisions=divisions,
            min_share=min_share,
        ) if figure else 0.0
        report["chance"] = round(chance, 3)

        if (
            figure
            and agreement >= min_agreement
            and evidence >= MIN_EVIDENCE
            and agreement >= chance + CHANCE_MARGIN
        ):
            rebuilt, _stats = apply_to_timeline(
                inside, figure, summary, start=edge, end=stop
            )
            result.extend(rebuilt)
            report["after"] = len(rebuilt)
            report["averaged"] = True
        else:
            result.extend(inside)

        reports.append(report)
        edge = stop

    result.sort(key=lambda n: (n.time, n.midi))
    return result, reports
