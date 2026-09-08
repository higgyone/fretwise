"""Re-listening to the least certain notes."""

import numpy as np
import pytest

from fretwise import review
from fretwise.analyze import DetectedNote
from fretwise.notes import midi_to_hz, midi_to_name, name_to_midi

SR = 22050


def note(name, time, confidence, *, duration=0.4, string=4, fret=0):
    midi = name_to_midi(name)
    made = DetectedNote(
        time=time, duration=duration, note=midi_to_name(midi), midi=midi,
        hz=midi_to_hz(midi), cents=0.0, confidence=confidence,
    )
    made.chosen = None if string is None else {"string": string, "fret": fret}
    return made


@pytest.fixture
def clip():
    return np.sin(2 * np.pi * 220 * np.arange(SR * 30) / SR).astype(np.float32) * 0.5


def test_the_weakest_notes_are_picked():
    notes = [note("D3", i, confidence=c) for i, c in enumerate([0.9, 0.2, 0.7, 0.1])]
    picked = review.weakest(notes, count=2)
    assert sorted(n.confidence for n in picked) == [0.1, 0.2]


def test_the_review_follows_the_recording_in_order():
    """Reviewing out of order is far harder to follow against the clip."""
    notes = [note("D3", i, confidence=c) for i, c in enumerate([0.9, 0.1, 0.7, 0.2])]
    picked = review.weakest(notes, count=2)
    assert [n.time for n in picked] == [1, 3]


def test_a_confidence_threshold_takes_everything_under_it():
    notes = [note("D3", i, confidence=c) for i, c in enumerate([0.9, 0.2, 0.7, 0.1])]
    picked = review.weakest(notes, below=0.5)
    assert [n.confidence for n in picked] == [0.2, 0.1]


def test_asking_for_none():
    notes = [note("D3", 0, confidence=0.5)]
    assert review.weakest(notes, count=0) == []
    assert review.weakest([], count=5) == []


def test_each_note_is_heard_twice(clip):
    """The recording alone, then the recording with the note over it."""
    audio, items = review.build(clip, SR, [note("D3", 5.0, 0.3)])
    assert len(items) == 1
    excerpt = 2 * review.PAD + 0.4
    expected = excerpt + review.GAP + excerpt + review.SEPARATION
    assert len(audio) / SR == pytest.approx(expected, abs=0.05)


def test_the_second_hearing_carries_the_detected_note(clip):
    """A 220Hz recording plus a detected A4 should show both pitches."""
    audio, _items = review.build(clip, SR, [note("A4", 5.0, 0.3)])
    excerpt = int((2 * review.PAD + 0.4) * SR)
    alone = audio[:excerpt]
    together = audio[excerpt + int(review.GAP * SR):][:excerpt]

    def level(signal, hz):
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
        freqs = np.fft.rfftfreq(len(signal), 1 / SR)
        return float(spectrum[np.argmin(np.abs(freqs - hz))])

    # The recording's own pitch is in both; the detected note only in the second.
    assert level(alone, 220) > 0
    assert level(together, 440) > 20 * level(alone, 440)


def test_a_very_short_note_still_gets_an_audible_reference(clip):
    """A tenth of a second is too brief to hear as a pitch."""
    audio, _items = review.build(clip, SR, [note("D3", 5.0, 0.3, duration=0.08)])
    excerpt = 2 * review.PAD + 0.08
    expected = excerpt + review.GAP + excerpt + review.SEPARATION
    assert len(audio) / SR == pytest.approx(expected, abs=0.05)
    assert review.MIN_TONE > 0.3


def test_items_say_where_to_listen(clip):
    notes = [note("D3", 5.0, 0.3), note("A3", 12.0, 0.2)]
    _audio, items = review.build(clip, SR, notes)
    assert items[0].at == 0.0
    assert items[1].at > items[0].at
    assert [item.time for item in items] == [5.0, 12.0]


def test_a_note_at_the_very_start_is_not_padded_past_the_beginning(clip):
    audio, items = review.build(clip, SR, [note("D3", 0.05, 0.3)])
    assert len(items) == 1
    assert len(audio) > 0


def test_a_note_beyond_the_recording_is_skipped(clip):
    _audio, items = review.build(clip, SR, [note("D3", 999.0, 0.3)])
    assert items == []


def test_a_quiet_passage_is_brought_up_to_be_audible():
    """Low confidence and low volume arrive together; a faint excerpt is useless."""
    quiet = np.sin(2 * np.pi * 220 * np.arange(SR * 10) / SR).astype(np.float32) * 0.002
    audio, _items = review.build(quiet, SR, [note("D3", 5.0, 0.3)])
    assert np.abs(audio).max() > 0.5


def test_nothing_to_review(clip):
    audio, items = review.build(clip, SR, [])
    assert items == [] and audio.size == 0
    assert "nothing to review" in review.index([])


def test_the_index_gives_both_clocks_and_the_position():
    notes = [note("D3", 65.0, 0.31, string=4, fret=0)]
    _audio, items = review.build(np.ones(SR * 80, dtype=np.float32), SR, notes)
    text = review.index(items)
    assert "1:05.000" in text  # where it is in the clip
    assert "0:00.000" in text  # where it is in the review
    assert "string 4 fret 0" in text
    assert "0.31" in text


def test_the_index_says_when_a_note_has_no_position():
    notes = [note("D3", 5.0, 0.31, string=None)]
    _audio, items = review.build(np.ones(SR * 20, dtype=np.float32), SR, notes)
    assert "unplaced" in review.index(items)
