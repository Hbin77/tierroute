# SPDX-License-Identifier: Apache-2.0
"""Large-integer text and identity encodings without process-global limit changes.

CPython deliberately limits direct decimal ``int`` conversion by default. Leaves stay
below even its minimum configurable threshold, while divide-and-conquer composition
avoids the quadratic behavior of repeated base-10 chunk accumulation and division.
"""

from __future__ import annotations

_LEAF_DIGITS = 600
_LOG10_2_NUMERATOR = 301_029_995_663
_LOG10_2_DENOMINATOR = 1_000_000_000_000


def _power_of_ten(width: int, powers: dict[int, int]) -> int:
    try:
        return powers[width]
    except KeyError:
        value = 10**width
        powers[width] = value
        return value


def _parse_unsigned_decimal(digits: str, powers: dict[int, int]) -> int:
    if len(digits) <= _LEAF_DIGITS:
        return int(digits)
    split = len(digits) // 2
    right_width = len(digits) - split
    return _parse_unsigned_decimal(digits[:split], powers) * _power_of_ten(
        right_width, powers
    ) + _parse_unsigned_decimal(digits[split:], powers)


def _decimal_width(value: int, powers: dict[int, int]) -> int:
    # This fixed-point approximation is below log10(2) by less than 1e-12.
    # Exact comparisons correct the estimate, keeping correctness independent of
    # floating-point rounding even for integers much larger than policy artifacts.
    width = ((value.bit_length() - 1) * _LOG10_2_NUMERATOR // _LOG10_2_DENOMINATOR) + 1
    boundary = _power_of_ten(width, powers)
    while value >= boundary:
        width += 1
        boundary *= 10
        powers.setdefault(width, boundary)
    while width > 1 and value < boundary // 10:
        width -= 1
        boundary //= 10
        powers.setdefault(width, boundary)
    return width


def _render_fixed_decimal(value: int, width: int, powers: dict[int, int]) -> str:
    if width <= _LEAF_DIGITS:
        return str(value).zfill(width)
    lower_width = width // 2
    upper, lower = divmod(value, _power_of_ten(lower_width, powers))
    return _render_fixed_decimal(upper, width - lower_width, powers) + _render_fixed_decimal(
        lower, lower_width, powers
    )


def integer_to_decimal(value: int) -> str:
    """Render an arbitrary integer without Python's decimal conversion digit cap."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if value == 0:
        return "0"
    negative = value < 0
    magnitude = -value if negative else value
    powers: dict[int, int] = {0: 1}
    width = _decimal_width(magnitude, powers)
    document = _render_fixed_decimal(magnitude, width, powers)
    return ("-" if negative else "") + document


def decimal_to_integer(value: str) -> int:
    """Parse ASCII decimal digits in bounded chunks without changing interpreter state."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if not digits or any(character < "0" or character > "9" for character in digits):
        raise ValueError("value must contain only an optional minus and decimal digits")
    result = _parse_unsigned_decimal(digits, {0: 1})
    return -result if negative else result


def integer_identity_bytes(value: int) -> bytes:
    """Return an unambiguous signed big-endian identity for hashing."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    magnitude = abs(value)
    width = max(1, (magnitude.bit_length() + 7) // 8)
    payload = magnitude.to_bytes(width, "big")
    return (b"\x01" if value < 0 else b"\x00") + width.to_bytes(8, "big") + payload
