"""Key estimation and scale degrees."""

import pytest

from fretwise.analyze import DetectedNote
from fretwise.key import (
    Key,
    annotate,
    degree_of,
    estimate_key,
    in_key_fraction,
    pitch_class_weights,
)
from fretwise.notes import name_to_midi


def note(name, duration=1.0, confidence=1.0):
    midi = name_to_midi(name)
    return DetectedNote(
        time=0.0, duration=duration, note=name, midi=midi,
        hz=440.0, cents=0.0, confidence=confidence,
    )


def notes(*names, **kwargs):
    return [note(name, **kwargs) for name in names]


C_MAJOR = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
A_MINOR = ["A3", "B3", "C4", "D4", "E4", "F4", "G4"]


def test_finds_a_major_key():
    # A C major scale with the tonic and dominant emphasised.
    played = notes(*C_MAJOR) + notes("C4", "C4", "G4")
    key = estimate_key(played)
    assert key.name == "C major"
    assert key.fit > 0.7


def test_finds_a_minor_key():
    played = notes(*A_MINOR) + notes("A3", "A3", "E4")
    key = estimate_key(played)
    assert key.name == "A minor"


def test_transposes():
    """The same intervals a fifth up should give the key a fifth up."""
    played = notes("G4", "A4", "B4", "C5", "D5", "E5", "F#5") + notes("G4", "G4", "D5")
    assert estimate_key(played).name == "G major"


def test_no_notes_gives_no_key():
    assert estimate_key([]) is None


def test_a_single_pitch_does_not_crash():
    assert estimate_key(notes("E4")) is not None


def test_weighting_favours_long_confident_notes():
    """A held, confident note counts for more than a brief uncertain one."""
    weights = pitch_class_weights(
        [note("C4", duration=4.0, confidence=1.0), note("F#4", duration=0.1, confidence=0.2)]
    )
    assert weights[0] > weights[6] * 100


@pytest.mark.parametrize(
    "name,degree,numeral",
    [
        ("C4", "1", "I"), ("D4", "2", "ii"), ("E4", "3", "iii"), ("F4", "4", "IV"),
        ("G4", "5", "V"), ("A4", "6", "vi"), ("B4", "7", "vii*"),
    ],
)
def test_major_degrees(name, degree, numeral):
    key = Key(tonic=0, mode="major", fit=1.0, margin=0.1)
    assert degree_of(name_to_midi(name), key) == (degree, numeral)


@pytest.mark.parametrize(
    "name,degree,numeral",
    [
        ("A3", "1", "i"), ("B3", "2", "ii*"), ("C4", "3", "III"), ("D4", "4", "iv"),
        ("E4", "5", "v"), ("F4", "6", "VI"), ("G4", "7", "VII"),
    ],
)
def test_minor_degrees(name, degree, numeral):
    key = Key(tonic=9, mode="minor", fit=1.0, margin=0.1)
    assert degree_of(name_to_midi(name), key) == (degree, numeral)


def test_chromatic_notes_get_an_accidental_and_no_numeral():
    key = Key(tonic=0, mode="major", fit=1.0, margin=0.1)
    assert degree_of(name_to_midi("D#4"), key) == ("b3", None)
    assert degree_of(name_to_midi("F#4"), key) == ("b5", None)


def test_degrees_are_octave_independent():
    key = Key(tonic=0, mode="major", fit=1.0, margin=0.1)
    assert degree_of(name_to_midi("G2"), key) == degree_of(name_to_midi("G6"), key)


def test_annotate_fills_in_every_note():
    key = Key(tonic=0, mode="major", fit=1.0, margin=0.1)
    played = annotate(notes("C4", "E4", "G4"), key)
    assert [n.degree for n in played] == ["1", "3", "5"]
    assert [n.numeral for n in played] == ["I", "iii", "V"]


def test_annotate_with_no_key_clears_the_fields():
    played = annotate(notes("C4"), None)
    assert played[0].degree is None and played[0].numeral is None


def test_in_key_fraction():
    key = Key(tonic=0, mode="major", fit=1.0, margin=0.1)
    assert in_key_fraction(notes(*C_MAJOR), key) == pytest.approx(1.0)
    assert in_key_fraction(notes("C4", "C#4"), key) == pytest.approx(0.5)
    assert in_key_fraction([], key) == 0.0


def test_degrees_appear_in_the_serialised_note():
    key = Key(tonic=0, mode="major", fit=1.0, margin=0.1)
    entry = annotate(notes("G4"), key)[0].to_dict()
    assert entry["degree"] == "5" and entry["numeral"] == "V"
