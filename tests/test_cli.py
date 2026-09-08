"""Smoke tests for command dispatch — every subcommand must reach its handler."""

import json
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from fretwise import cli
from test_analyze import SR, melody


@pytest.fixture
def work_dir(tmp_path):
    sf.write(tmp_path / "clip.wav", melody(["E4", "G4", "B4"], 0.4, 0.08), SR)
    return tmp_path


def test_every_subcommand_has_a_handler():
    """Guards against a parser knowing a command that main() cannot dispatch."""
    parser = cli.build_parser()
    commands = parser._subparsers._group_actions[0].choices
    for name in commands:
        assert hasattr(cli, f"run_{name}"), f"no run_{name}() for `fretwise {name}`"


def test_analyze_writes_notes_json(work_dir, capsys):
    assert cli.main(["analyze", "-o", str(work_dir)]) == 0
    payload = json.loads((work_dir / "notes.json").read_text())
    assert [n["note"] for n in payload["notes"]] == ["E4", "G4", "B4"]
    assert "3 notes" in capsys.readouterr().out


def test_analyze_reports_a_key_and_degrees(work_dir, capsys):
    """An E minor triad should be labelled with degrees of some key."""
    assert cli.main(["analyze", "-o", str(work_dir)]) == 0
    payload = json.loads((work_dir / "notes.json").read_text())

    assert payload["key"]["mode"] in {"major", "minor"}
    assert 0 <= payload["key"]["in_key"] <= 1
    for entry in payload["notes"]:
        assert entry["degree"] is not None
    assert "key:" in capsys.readouterr().out


def test_analyze_without_a_clip_is_an_error(tmp_path, capsys):
    assert cli.main(["analyze", "-o", str(tmp_path)]) == 1
    assert "run `fretwise ingest` first" in capsys.readouterr().err


def test_analyze_rejects_a_bad_pitch_bound(work_dir, capsys):
    assert cli.main(["analyze", "-o", str(work_dir), "--fmin", "banana"]) == 1
    assert "not a note name or frequency" in capsys.readouterr().err


def test_ingest_reports_the_clip(monkeypatch, tmp_path, capsys):
    """Dispatch and output formatting, without touching the network."""
    from fretwise.ingest import Clip

    recorded = {}

    def fake_ingest(url, **kwargs):
        recorded.update(url=url, **kwargs)
        return Clip(
            path=tmp_path / "clip.wav", source_url=url, title="A Song",
            start=90.0, end=105.0, sample_rate=44100,
        )

    monkeypatch.setattr(cli, "ingest", fake_ingest)
    assert cli.main(["ingest", "http://example.com/v", "--start", "1:30", "--end", "1:45"]) == 0

    assert recorded["url"] == "http://example.com/v"
    assert (recorded["start"], recorded["end"]) == ("1:30", "1:45")
    output = capsys.readouterr().out
    assert "A Song" in output
    assert "1:30.000" in output and "15.00s" in output


def test_analyze_rejects_an_unknown_stem_name(work_dir, capsys):
    """--stems names must be stems, not typos, and say so before doing work."""
    assert cli.main(["analyze", "-o", str(work_dir), "--stems", "guitar,banjo"]) == 1
    assert "unknown stem" in capsys.readouterr().err


def test_analyze_says_which_stems_are_missing(work_dir, capsys):
    """Summing stems needs them separated first."""
    assert cli.main(["analyze", "-o", str(work_dir), "--stems", "guitar,bass"]) == 1
    assert "separate --all" in capsys.readouterr().err
