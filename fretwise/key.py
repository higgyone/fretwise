"""Key estimation and scale degrees.

Given the notes found in a clip, guess which key they are in and label each
note with its degree in that key -- both as a number (``1``, ``b3``, ``5``)
and, for diatonic degrees, as the Roman numeral of the triad built on it
(``I ii iii IV V vi vii*``).

Key estimation uses the Krumhansl-Kessler profiles: a weighted histogram of
pitch classes is correlated against a major and a minor template rotated to
all twelve tonics, and the best correlation wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notes import NAMES_SHARP

# Krumhansl-Kessler key profiles: the perceived stability of each scale
# degree, from listener ratings. Index 0 is the tonic.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Semitone offsets of each scale degree from the tonic.
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)

# Triad quality per degree, as Roman numeral case. "*" marks a diminished triad.
MAJOR_NUMERALS = ("I", "ii", "iii", "IV", "V", "vi", "vii*")
MINOR_NUMERALS = ("i", "ii*", "III", "iv", "v", "VI", "VII")

# Names for pitches that fall outside the scale, by semitone offset from the tonic.
MAJOR_CHROMATIC = {1: "b2", 3: "b3", 6: "b5", 8: "b6", 10: "b7"}
MINOR_CHROMATIC = {1: "b2", 4: "3", 6: "b5", 9: "6", 11: "7"}


@dataclass(frozen=True)
class Key:
    """An estimated key."""

    tonic: int  # pitch class, 0 = C
    mode: str  # "major" or "minor"
    fit: float  # correlation with the profile, -1..1
    margin: float  # how far ahead of the runner-up key

    @property
    def name(self) -> str:
        return f"{NAMES_SHARP[self.tonic]} {self.mode}"

    @property
    def steps(self) -> tuple[int, ...]:
        return MAJOR_STEPS if self.mode == "major" else MINOR_STEPS

    @property
    def numerals(self) -> tuple[str, ...]:
        return MAJOR_NUMERALS if self.mode == "major" else MINOR_NUMERALS

    def __str__(self) -> str:
        return self.name


def pitch_class_weights(notes) -> np.ndarray:
    """Histogram of pitch classes, weighted by how long and how clearly played.

    Weighting by duration keeps a held tonic from counting the same as a
    passing note, and weighting by confidence keeps shaky detections from
    swinging the estimate.
    """
    weights = np.zeros(12)
    for note in notes:
        weights[note.midi % 12] += max(note.duration, 0.0) * max(note.confidence, 0.0)
    return weights


def estimate_key(notes) -> Key | None:
    """Best-fitting key for a set of notes, or ``None`` if there is nothing to go on."""
    weights = pitch_class_weights(notes)
    if not weights.any():
        return None

    scored: list[tuple[float, int, str]] = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            rotated = np.roll(profile, tonic)
            correlation = np.corrcoef(weights, rotated)[0, 1]
            if np.isnan(correlation):  # a single pitch class has no variance
                correlation = 0.0
            scored.append((float(correlation), tonic, mode))

    scored.sort(reverse=True)
    best, runner_up = scored[0], scored[1]
    return Key(
        tonic=best[1], mode=best[2], fit=best[0], margin=best[0] - runner_up[0]
    )


def degree_of(midi: int, key: Key) -> tuple[str, str | None]:
    """Scale degree of a note in ``key``.

    Returns ``(degree, numeral)``. ``degree`` is always present -- notes
    outside the scale get an accidental, e.g. ``b3``. ``numeral`` is the Roman
    numeral of the triad on that degree, or ``None`` for a chromatic note,
    which does not have one.
    """
    offset = (midi - key.tonic) % 12
    if offset in key.steps:
        index = key.steps.index(offset)
        return str(index + 1), key.numerals[index]
    chromatic = MAJOR_CHROMATIC if key.mode == "major" else MINOR_CHROMATIC
    return chromatic[offset], None


def annotate(notes, key: Key | None) -> list:
    """Fill in each note's ``degree`` and ``numeral`` for ``key``."""
    for note in notes:
        if key is None:
            note.degree, note.numeral = None, None
        else:
            note.degree, note.numeral = degree_of(note.midi, key)
    return notes


def in_key_fraction(notes, key: Key) -> float:
    """Share of played time whose pitch belongs to ``key``'s scale.

    A useful sanity check on an estimate: seven of twelve pitch classes are
    diatonic, so pitches scattered at random score about 0.58.
    """
    weights = pitch_class_weights(notes)
    total = weights.sum()
    if total <= 0:
        return 0.0
    diatonic = sum(weights[(key.tonic + step) % 12] for step in key.steps)
    return float(diatonic / total)
