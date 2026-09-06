"""Stages 3 and 4 — onset detection and pitch tracking.

Finds where each note starts, estimates the fundamental frequency over the
segment that follows, and snaps it to a named pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np

from .notes import midi_to_hz, name_to_midi, snap

# Default search range: the open low E of a guitar up to the 24th fret of the
# high E. pyin will not report a pitch sitting on the boundary itself, so the
# range is padded by a semitone at each end to keep E2 and E6 reachable.
RANGE_PADDING_SEMITONES = 1
DEFAULT_FMIN = midi_to_hz(name_to_midi("E2") - RANGE_PADDING_SEMITONES)
DEFAULT_FMAX = midi_to_hz(name_to_midi("E6") + RANGE_PADDING_SEMITONES)

DEFAULT_HOP_LENGTH = 256
# Ignore the first slice of each segment: pick attacks are broadband noise and
# confuse the pitch tracker before the string settles.
ATTACK_SKIP = 0.03
MIN_DURATION = 0.05
# pyin's voiced probability is near 1.0 on a clean synthesised tone but runs an
# order of magnitude lower on real recordings, so this is calibrated against
# real audio: on a separated guitar stem the pitched segments sit around
# 0.15-0.25, while a drum stem (nothing pitched to find) stays below 0.01 and
# yields no notes at all at this threshold. Raise it to trade recall for
# precision.
MIN_CONFIDENCE = 0.1


@dataclass
class DetectedNote:
    """One note found in the clip."""

    time: float
    duration: float
    note: str
    midi: int
    hz: float
    cents: float
    confidence: float
    # Filled in by fretwise.key.annotate once the key is known.
    degree: str | None = None
    numeral: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "DetectedNote":
        return cls(**{field: data[field] for field in cls.__dataclass_fields__ if field in data})

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("time", "duration", "hz", "cents", "confidence"):
            data[key] = round(data[key], 4)
        return data


def load_audio(path: Path | str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load a clip as a mono float array."""
    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return y, sr


def detect_onsets(
    y: np.ndarray,
    sr: int,
    *,
    sensitivity: float = 0.5,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Onset times in seconds.

    ``sensitivity`` runs 0..1: higher finds more onsets (and more false
    positives). It maps inversely onto librosa's peak-picking ``delta``.
    """
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(f"sensitivity must be in 0..1, got {sensitivity}")
    if y.size == 0:
        return np.array([])

    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    delta = 0.02 + (1.0 - sensitivity) * 0.28
    return librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop_length,
        units="time",
        backtrack=True,
        delta=delta,
    )


def segment_bounds(onsets: np.ndarray, duration: float) -> list[tuple[float, float]]:
    """Turn onset times into ``(start, end)`` spans running up to the next onset."""
    times = [t for t in np.atleast_1d(onsets) if t < duration]
    ends = list(times[1:]) + [duration]
    return [(start, end) for start, end in zip(times, ends) if end > start]


def track_pitch(
    y: np.ndarray,
    sr: int,
    *,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pyin over the whole clip. Returns ``(times, f0, voiced_prob)``."""
    f0, _voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_prob


@dataclass
class Candidate:
    """One onset, with what was found there and whether it survived filtering.

    Every onset produces a candidate, including the rejected ones. Keeping
    the rejects visible is the only way to tell a note the detector never
    saw from one it saw and threw away.
    """

    time: float
    duration: float
    confidence: float
    voiced_fraction: float
    verdict: str  # "kept", "short", "unvoiced", or "low-confidence"
    hz: float | None = None
    midi: int | None = None
    note: str | None = None
    cents: float | None = None

    @property
    def kept(self) -> bool:
        return self.verdict == "kept"

    def to_note(self) -> DetectedNote:
        return DetectedNote(
            time=self.time, duration=self.duration, note=self.note, midi=self.midi,
            hz=self.hz, cents=self.cents, confidence=self.confidence,
        )

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 4),
            "duration": round(self.duration, 4),
            "note": self.note,
            "midi": self.midi,
            "hz": None if self.hz is None else round(self.hz, 4),
            "cents": None if self.cents is None else round(self.cents, 4),
            "confidence": round(self.confidence, 4),
            "voiced_fraction": round(self.voiced_fraction, 4),
            "verdict": self.verdict,
        }


def _summarize_segment(
    start: float,
    end: float,
    times: np.ndarray,
    f0: np.ndarray,
    voiced_prob: np.ndarray,
    *,
    attack_skip: float = ATTACK_SKIP,
) -> tuple[float, float, float] | None:
    """Pitch, confidence and voiced fraction over one segment.

    Returns ``None`` when no frame in the segment carried a pitch at all.
    """
    window_start = start + attack_skip
    if window_start >= end:  # very short segment: use all of it rather than nothing
        window_start = start

    in_window = (times >= window_start) & (times < end)
    voiced = in_window & np.isfinite(f0)
    if not voiced.any():
        return None

    voiced_fraction = float(voiced.sum() / max(int(in_window.sum()), 1))
    return (
        float(np.median(f0[voiced])),
        float(np.mean(voiced_prob[voiced])),
        voiced_fraction,
    )


def candidates(
    y: np.ndarray,
    sr: int,
    *,
    sensitivity: float = 0.5,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    hop_length: int = DEFAULT_HOP_LENGTH,
    min_duration: float = MIN_DURATION,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Candidate]:
    """Every onset with its pitch estimate and the verdict passed on it."""
    duration = len(y) / sr
    onsets = detect_onsets(y, sr, sensitivity=sensitivity, hop_length=hop_length)
    if len(onsets) == 0:
        return []

    times, f0, voiced_prob = track_pitch(
        y, sr, fmin=fmin, fmax=fmax, hop_length=hop_length
    )

    found: list[Candidate] = []
    for start, end in segment_bounds(onsets, duration):
        span = end - start
        if span < min_duration:
            found.append(Candidate(start, span, 0.0, 0.0, "short"))
            continue

        summary = _summarize_segment(start, end, times, f0, voiced_prob)
        if summary is None:
            found.append(Candidate(start, span, 0.0, 0.0, "unvoiced"))
            continue

        hz, confidence, voiced_fraction = summary
        midi, name, cents = snap(hz)
        verdict = "kept" if confidence >= min_confidence else "low-confidence"
        found.append(
            Candidate(
                time=start, duration=span, confidence=confidence,
                voiced_fraction=voiced_fraction, verdict=verdict,
                hz=hz, midi=midi, note=name, cents=cents,
            )
        )
    return found


def analyze(y: np.ndarray, sr: int, **kwargs) -> list[DetectedNote]:
    """Run stages 3-5 over loaded audio and return the notes that survived."""
    return [c.to_note() for c in candidates(y, sr, **kwargs) if c.kept]


def load_notes(path: Path | str) -> tuple[list[DetectedNote], dict | None]:
    """Read a notes.json written by the analyze stage. Returns ``(notes, key)``."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DetectedNote.from_dict(entry) for entry in payload["notes"]], payload.get("key")


def analyze_file(path: Path | str, **kwargs) -> list[DetectedNote]:
    """Load a clip from disk and analyze it."""
    y, sr = load_audio(path)
    return analyze(y, sr, **kwargs)


def candidates_file(path: Path | str, **kwargs) -> list[Candidate]:
    """Load a clip from disk and report every onset in it."""
    y, sr = load_audio(path)
    return candidates(y, sr, **kwargs)
