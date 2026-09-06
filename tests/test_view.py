"""Building the practice page."""

import json
import re

import pytest

from fretwise.analyze import DetectedNote
from fretwise.notes import midi_to_hz, midi_to_name, name_to_midi
from fretwise.view import (
    DEFAULT_VIEW_SPEED,
    ViewError,
    build_payload,
    escape,
    find_speeds,
    note_payload,
    render_page,
    write_page,
)


def note(name, time=0.0, duration=0.5, string=4, fret=0):
    midi = name_to_midi(name)
    made = DetectedNote(
        time=time, duration=duration, note=midi_to_name(midi), midi=midi,
        hz=midi_to_hz(midi), cents=0.0, confidence=0.8,
    )
    made.degree, made.numeral = "1", "I"
    made.chosen = None if string is None else {"string": string, "fret": fret}
    return made


def embedded(html):
    """Pull the JSON back out of a built page, as the page's own script does."""
    block = re.search(r'type="application/json">(.*?)</script>', html, re.S).group(1)
    return json.loads(block.replace("<\\/", "</"))


@pytest.fixture
def work(tmp_path):
    for name in ("clip-50.wav", "clip-75.wav", "clip-100.wav"):
        (tmp_path / name).write_bytes(b"RIFF")  # presence is all that is checked
    return tmp_path


def test_speeds_survive_the_trip_to_javascript(work):
    """JSON writes 1.0 as 1, so speeds are a list, not float-keyed."""
    page = write_page([note("D3")], work)
    data = embedded(page.path.read_text(encoding="utf-8"))

    assert isinstance(data["speeds"], list)
    assert [entry["speed"] for entry in data["speeds"]] == [0.5, 0.75, 1.0]
    assert data["speeds"][2] == {"speed": 1.0, "file": "clip-100.wav"}
    # Every speed offered must be findable by exact value, as the page does.
    for entry in data["speeds"]:
        assert any(e["speed"] == entry["speed"] for e in data["speeds"])


def test_the_default_speed_is_one_that_exists(work):
    (work / "clip-75.wav").unlink()
    page = write_page([note("D3")], work, default_speed=0.75)
    data = embedded(page.path.read_text(encoding="utf-8"))
    assert data["defaultSpeed"] in [entry["speed"] for entry in data["speeds"]]


def test_missing_audio_is_reported_not_offered(work):
    (work / "clip-50.wav").unlink()
    page = write_page([note("D3")], work)
    assert page.missing_audio == ["clip-50.wav"]
    assert 0.5 not in page.speeds


def test_no_audio_at_all_is_an_error(tmp_path):
    with pytest.raises(ViewError, match="run `fretwise render`"):
        write_page([note("D3")], tmp_path)


def test_note_payload_flattens_the_chosen_position():
    payload = note_payload(note("D3", string=4, fret=0))
    assert payload["string"] == 4 and payload["fret"] == 0
    assert payload["degree"] == "1" and payload["numeral"] == "I"


def test_a_note_with_no_position_still_serialises():
    """An unplaceable note must not break the page; it just has no dot."""
    payload = note_payload(note("D3", string=None))
    assert payload["string"] is None and payload["fret"] is None


def test_pitch_class_drops_the_octave():
    """Dots are labelled with the pitch class; the neck implies the octave."""
    assert note_payload(note("F#3"))["pitchClass"] == "F#"
    assert note_payload(note("C-1"))["pitchClass"] == "C"


def test_duration_covers_the_last_note():
    notes = [note("D3", time=0.0, duration=1.0), note("A3", time=10.0, duration=2.5)]
    payload = build_payload(
        notes, key=None, speeds={1.0: "clip-100.wav"}, default_speed=1.0
    )
    assert payload["duration"] == pytest.approx(12.5)


def test_duration_is_never_zero():
    """A zero span would divide by zero when placing the playhead."""
    payload = build_payload(
        [], key=None, speeds={1.0: "clip-100.wav"}, default_speed=1.0
    )
    assert payload["duration"] > 0


def test_the_page_carries_its_data_and_needs_no_server(work):
    page = write_page([note("D3"), note("A3", time=1.0)], work, title="A Song")
    html = page.path.read_text(encoding="utf-8")
    assert "A Song" in html
    assert len(embedded(html)["notes"]) == 2
    # Nothing is fetched: only the audio files are referenced by name.
    assert "fetch(" not in html


def test_a_closing_script_tag_in_the_data_cannot_break_out():
    """A title or note field must not be able to end the data block early."""
    payload = {"notes": [{"note": "</script><script>alert(1)</script>"}]}
    html = render_page(payload, title="x")
    assert "</script><script>alert" not in html
    assert embedded(html)["notes"][0]["note"] == "</script><script>alert(1)</script>"


def test_the_title_is_escaped():
    assert escape('<img src=x onerror="bad">') == (
        "&lt;img src=x onerror=&quot;bad&quot;&gt;"
    )


def test_find_speeds_separates_present_from_missing(tmp_path):
    (tmp_path / "clip-75.wav").write_bytes(b"RIFF")
    present, missing = find_speeds(tmp_path)
    assert present == {0.75: "clip-75.wav"}
    assert missing == ["clip-50.wav", "clip-100.wav"]


def test_the_default_view_speed_is_slower_than_full():
    """Practising happens slowly, so the page should not open at full speed."""
    assert DEFAULT_VIEW_SPEED < 1.0


def test_tuning_and_fret_count_reach_the_page(work):
    page = write_page([note("D3")], work, max_fret=15)
    data = embedded(page.path.read_text(encoding="utf-8"))
    assert data["maxFret"] == 15
    assert data["tuning"][0] == "E4" and data["tuning"][5] == "E2"


def test_the_page_says_so_when_audio_is_missing(work):
    """Without its .wav files the page looked broken rather than incomplete."""
    html = write_page([note("D3")], work).path.read_text(encoding="utf-8")
    assert 'id="warn"' in html
    assert "Audio not found" in html


def test_the_page_still_runs_without_audio(work):
    """A clock of its own drives the fretboard when the audio cannot load."""
    html = write_page([note("D3")], work).path.read_text(encoding="utf-8")
    assert "audioUsable" in html
    assert "silentPlaying" in html
    # The display reads the clock through one accessor, not audio directly.
    assert "function currentTime()" in html
