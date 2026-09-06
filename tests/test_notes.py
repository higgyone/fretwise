import pytest

from fretwise.notes import (
    NoteError,
    hz_to_midi,
    midi_to_hz,
    midi_to_name,
    name_to_midi,
    snap,
)


@pytest.mark.parametrize(
    "hz,midi",
    [(440.0, 69), (82.4069, 40), (329.628, 64), (1318.51, 88), (261.626, 60)],
)
def test_hz_midi_roundtrip(hz, midi):
    assert hz_to_midi(hz) == pytest.approx(midi, abs=0.01)
    assert midi_to_hz(midi) == pytest.approx(hz, rel=1e-4)


@pytest.mark.parametrize(
    "midi,name",
    [(40, "E2"), (60, "C4"), (64, "E4"), (69, "A4"), (88, "E6"), (61, "C#4"), (0, "C-1")],
)
def test_names(midi, name):
    assert midi_to_name(midi) == name
    assert name_to_midi(name) == midi


@pytest.mark.parametrize("name,midi", [("Bb3", 58), ("eb4", 63), ("e4", 64)])
def test_name_accepts_flats_and_lowercase(name, midi):
    assert name_to_midi(name) == midi


@pytest.mark.parametrize("name", ["H4", "E", "E#4x", "", "4E"])
def test_name_rejects_junk(name):
    with pytest.raises(NoteError):
        name_to_midi(name)


def test_hz_must_be_positive():
    with pytest.raises(NoteError):
        hz_to_midi(0)


def test_snap_reports_cents_offset():
    midi, name, cents = snap(440.0)
    assert (midi, name) == (69, "A4")
    assert cents == pytest.approx(0.0, abs=1e-9)

    # A quarter-tone sharp of A4 is still A4, ~50 cents off.
    midi, name, cents = snap(440.0 * 2 ** (0.49 / 12))
    assert name == "A4"
    assert cents == pytest.approx(49, abs=0.5)

    # 20 cents flat of E4 snaps to E4.
    midi, name, cents = snap(midi_to_hz(64 - 0.2))
    assert name == "E4"
    assert cents == pytest.approx(-20, abs=0.5)
