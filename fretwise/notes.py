"""Stage 5 — note mapping.

Conversions between frequency, MIDI number and scientific pitch names
(``E4``). Pitch names use sharps; A4 = 440 Hz.
"""

from __future__ import annotations

import math
import re

A4_HZ = 440.0
A4_MIDI = 69

NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
_NOTE_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


class NoteError(ValueError):
    """Raised when a note name cannot be parsed."""


def hz_to_midi(hz: float) -> float:
    """Fractional MIDI number for a frequency in Hz."""
    if hz <= 0:
        raise NoteError(f"frequency must be positive, got {hz}")
    return A4_MIDI + 12 * math.log2(hz / A4_HZ)


def midi_to_hz(midi: float) -> float:
    """Frequency in Hz for a (possibly fractional) MIDI number."""
    return A4_HZ * 2 ** ((midi - A4_MIDI) / 12)


def midi_to_name(midi: int) -> str:
    """Scientific pitch name for an integer MIDI number, e.g. 64 -> ``E4``."""
    midi = int(midi)
    return f"{NAMES_SHARP[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name: str) -> int:
    """Inverse of :func:`midi_to_name`. Accepts flats (``Bb3``) too."""
    match = _NOTE_PATTERN.match(name.strip())
    if not match:
        raise NoteError(f"not a valid note name: {name!r}")
    letter, accidental, octave = match.groups()
    pitch_class = f"{letter.upper()}{accidental}"
    pitch_class = _FLAT_TO_SHARP.get(pitch_class, pitch_class)
    if pitch_class not in NAMES_SHARP:
        raise NoteError(f"not a valid note name: {name!r}")
    return NAMES_SHARP.index(pitch_class) + (int(octave) + 1) * 12


def snap(hz: float) -> tuple[int, str, float]:
    """Snap a frequency to the nearest semitone.

    Returns ``(midi, name, cents_off)`` where ``cents_off`` is how far the
    input sits from that semitone (negative = flat, range -50..+50).
    """
    exact = hz_to_midi(hz)
    midi = int(round(exact))
    return midi, midi_to_name(midi), (exact - midi) * 100
