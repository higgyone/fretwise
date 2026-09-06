"""Play back the notes that were detected.

Re-synthesises the note list as plucked tones at the times and pitches that
were found, so it can be listened to against the original clip. If the
detector is hearing the part correctly the two line up; where it is wrong,
the wrong note is immediately obvious to anyone who knows the song.

This is the ear-level check on the analysis stages, and it is much better at
finding gross errors than any confidence number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Relative amplitude of each harmonic. Falling off as 1/h**1.5 gives a plucked,
# slightly nasal tone that is easy to pick out against a real recording.
HARMONICS = (1, 2, 3, 4, 5)
DECAY = 3.5
RELEASE = 0.02
NOTE_GAIN = 0.7
CLIP_GAIN = 0.5


def synth_note(hz: float, duration: float, sr: int, *, decay: float = DECAY) -> np.ndarray:
    """One plucked tone: decaying harmonics, ramped to silence at the end."""
    length = max(int(duration * sr), 1)
    t = np.arange(length) / sr
    envelope = np.exp(-decay * t)

    ramp = min(int(RELEASE * sr), length)
    if ramp:
        envelope[-ramp:] *= np.linspace(1.0, 0.0, ramp)

    wave = sum((1.0 / h**1.5) * np.sin(2 * np.pi * hz * h * t) for h in HARMONICS)
    return (wave * envelope).astype(np.float32)


def click(sr: int, *, length: float = 0.012, hz: float = 2200.0) -> np.ndarray:
    """A short blip, for marking where an onset was detected."""
    t = np.arange(int(length * sr)) / sr
    return (np.sin(2 * np.pi * hz * t) * np.exp(-90.0 * t)).astype(np.float32)


def render_clicks(times, sr: int, duration: float) -> np.ndarray:
    """A click track marking every onset, whether or not it became a note."""
    output = np.zeros(max(int(duration * sr), 1), dtype=np.float32)
    blip = click(sr)
    for time in times:
        start = int(time * sr)
        end = min(start + len(blip), len(output))
        if start < len(output):
            output[start:end] += blip[: end - start]
    return normalize(output)


def render(notes, sr: int, duration: float | None = None) -> np.ndarray:
    """Render a note list to audio, each note at its own time and pitch."""
    if duration is None:
        duration = max((n.time + n.duration for n in notes), default=0.0)
    output = np.zeros(max(int(duration * sr), 1), dtype=np.float32)

    for note in notes:
        tone = synth_note(note.hz, note.duration, sr)
        start = int(note.time * sr)
        end = min(start + len(tone), len(output))
        if start < len(output):
            output[start:end] += tone[: end - start]

    return normalize(output)


def normalize(audio: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """Scale to a fixed peak, leaving overlapping notes intact rather than clipped."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 0:
        return audio
    return (audio * (headroom / peak)).astype(np.float32)


def mix(clip: np.ndarray, rendered: np.ndarray, *, clip_gain: float = CLIP_GAIN,
        note_gain: float = NOTE_GAIN) -> np.ndarray:
    """Lay the rendered notes over the original so alignment can be heard."""
    length = max(len(clip), len(rendered))
    output = np.zeros(length, dtype=np.float32)
    output[: len(clip)] += normalize(clip) * clip_gain
    output[: len(rendered)] += rendered * note_gain
    return normalize(output)


def write(audio: np.ndarray, sr: int, destination: Path | str) -> Path:
    import soundfile as sf

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio, sr)
    return destination
