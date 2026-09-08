"""Re-listening to the notes least worth trusting.

Confidence says which notes the transcription is least sure of, but a number
cannot say whether a note is right. This builds a short audio file that plays,
for each doubtful note, the recording around it followed by the note it was
read as -- so the two can be compared by ear, which is the only thing that
settles it.

The recording is normalised segment by segment, because low confidence and
low volume tend to arrive together and the passage in question is often the
quietest thing on the clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sonify import normalize, synth_note

# Heard either side of the note, for context: a note alone is hard to place.
PAD = 0.6
# Between the recording and the note it was read as.
GAP = 0.2
# After each pair, before the next.
SEPARATION = 0.7
# A reference tone longer than this outstays its welcome.
MAX_TONE = 1.2
DEFAULT_COUNT = 10


@dataclass
class Item:
    """One doubtful note, and where it sits in the review audio."""

    note: object
    at: float  # seconds into the review file

    @property
    def time(self) -> float:
        return self.note.time


def weakest(notes, *, count: int = DEFAULT_COUNT, below: float | None = None) -> list:
    """The notes least worth trusting, in the order they are played.

    ``below`` takes everything under a confidence; otherwise the ``count``
    weakest. Ordering the result by time keeps the review in step with the
    recording, which makes it far easier to follow.
    """
    if below is not None:
        picked = [n for n in notes if n.confidence < below]
    else:
        picked = sorted(notes, key=lambda n: n.confidence)[: max(count, 0)]
    return sorted(picked, key=lambda n: n.time)


def excerpt(clip: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """A slice of the recording, clamped to what exists."""
    first = max(0, int(start * sr))
    last = min(len(clip), int(end * sr))
    return clip[first:last] if last > first else np.zeros(0, dtype=np.float32)


def build(
    clip: np.ndarray,
    sr: int,
    notes,
    *,
    pad: float = PAD,
    gap: float = GAP,
    separation: float = SEPARATION,
) -> tuple[np.ndarray, list[Item]]:
    """Build the review audio. Returns the audio and where each note lands."""
    silence = lambda seconds: np.zeros(max(int(seconds * sr), 0), dtype=np.float32)

    pieces: list[np.ndarray] = []
    items: list[Item] = []
    position = 0.0

    for note in notes:
        heard = excerpt(clip, sr, note.time - pad, note.time + note.duration + pad)
        if not heard.size:
            continue
        # Each excerpt is levelled on its own: a doubtful note is often the
        # quietest passage on the clip, and would be inaudible beside the rest.
        heard = normalize(heard, headroom=0.85)
        tone = synth_note(note.hz, min(note.duration, MAX_TONE), sr) * 0.7

        items.append(Item(note=note, at=position))
        for piece in (heard, silence(gap), tone, silence(separation)):
            pieces.append(piece)
            position += len(piece) / sr

    if not pieces:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(pieces).astype(np.float32), items


def index(items: list[Item]) -> str:
    """A listing to read while the review plays."""
    from .timestamps import format_timestamp

    if not items:
        return "nothing to review\n"

    lines = [
        f"{'in review':>10}  {'in clip':>9}  {'note':<5} {'conf':>5}  where"
    ]
    for item in items:
        note = item.note
        where = (
            f"string {note.chosen['string']} fret {note.chosen['fret']}"
            if getattr(note, "chosen", None)
            else "unplaced"
        )
        lines.append(
            f"{format_timestamp(item.at):>10}  {format_timestamp(note.time):>9}  "
            f"{note.note:<5} {note.confidence:5.2f}  {where}"
        )
    return "\n".join(lines) + "\n"
