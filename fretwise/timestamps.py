"""Parsing and formatting of clip timestamps."""

from __future__ import annotations


class TimestampError(ValueError):
    """Raised when a timestamp string cannot be parsed."""


def parse_timestamp(value: str | float | int | None) -> float | None:
    """Parse ``SS``, ``MM:SS`` or ``HH:MM:SS`` (fractional seconds allowed).

    Returns seconds as a float, or ``None`` if ``value`` is ``None``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise TimestampError(f"timestamp cannot be negative: {value}")
        return seconds

    text = value.strip()
    if not text:
        return None

    parts = text.split(":")
    if len(parts) > 3:
        raise TimestampError(f"too many ':' separated fields: {value!r}")

    try:
        fields = [float(p) for p in parts]
    except ValueError:
        raise TimestampError(f"not a valid timestamp: {value!r}") from None

    if any(field < 0 for field in fields):
        raise TimestampError(f"timestamp fields cannot be negative: {value!r}")
    # Only the last field may be fractional; earlier ones must be whole minutes/hours.
    for field in fields[:-1]:
        if field != int(field):
            raise TimestampError(f"only the seconds field may be fractional: {value!r}")

    seconds = 0.0
    for field in fields:
        seconds = seconds * 60 + field
    if seconds < 0:
        raise TimestampError(f"timestamp cannot be negative: {value!r}")
    return seconds


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``H:MM:SS.mmm`` (hours dropped when zero)."""
    if seconds < 0:
        raise TimestampError(f"timestamp cannot be negative: {seconds}")
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{secs:06.3f}"
    return f"{int(minutes)}:{secs:06.3f}"
