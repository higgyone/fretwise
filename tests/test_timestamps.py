import pytest

from fretwise.timestamps import TimestampError, format_timestamp, parse_timestamp


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0.0),
        ("83", 83.0),
        ("1:23", 83.0),
        ("1:23.5", 83.5),
        ("01:02:03", 3723.0),
        ("  12  ", 12.0),
        ("", None),
        (None, None),
        (45, 45.0),
        (45.5, 45.5),
    ],
)
def test_parse(text, expected):
    assert parse_timestamp(text) == expected


@pytest.mark.parametrize("text", ["abc", "1:2:3:4", "1.5:00", "-3", "1:-2"])
def test_parse_rejects(text):
    with pytest.raises(TimestampError):
        parse_timestamp(text)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0:00.000"), (83.5, "1:23.500"), (3723, "1:02:03.000")],
)
def test_format(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_roundtrip():
    assert parse_timestamp(format_timestamp(3723.25)) == 3723.25
