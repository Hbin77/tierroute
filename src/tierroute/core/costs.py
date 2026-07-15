# SPDX-License-Identifier: Apache-2.0
"""Context-independent arithmetic for exact decimal costs.

``Decimal`` operators obey the caller's mutable global context. That is useful for
general numerical work, but unsafe at a budget boundary: a low-precision context can
round a small overspend away. These helpers operate on decimal coefficient/exponent
tuples and Python integers, so finite additions, subtractions, and integer scaling do
not round and do not depend on ambient process state.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, Context, Decimal, localcontext
from math import gcd

from tierroute.core.schemas import Cost


def _parts(value: Cost, field_name: str) -> tuple[int, int]:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"{field_name} must be finite")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if sign:
        coefficient = -coefficient
    return coefficient, exponent


def _from_parts(coefficient: int, exponent: int) -> Cost:
    if coefficient == 0:
        return Decimal(0)
    sign = int(coefficient < 0)
    coefficient = abs(coefficient)

    # A canonical result avoids retaining redundant zero digits after repeated sums.
    while coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1

    reversed_digits: list[int] = []
    while coefficient:
        coefficient, digit = divmod(coefficient, 10)
        reversed_digits.append(digit)
    return Decimal((sign, tuple(reversed(reversed_digits)), exponent))


def add_cost(left: Cost, right: Cost) -> Cost:
    """Return the exact sum of two non-negative finite costs."""

    left_coefficient, left_exponent = _parts(left, "left")
    right_coefficient, right_exponent = _parts(right, "right")
    if left_coefficient == 0:
        return right
    if right_coefficient == 0:
        return left
    common_exponent = min(left_exponent, right_exponent)
    coefficient = left_coefficient * 10 ** (left_exponent - common_exponent)
    coefficient += right_coefficient * 10 ** (right_exponent - common_exponent)
    return _from_parts(coefficient, common_exponent)


def subtract_cost(left: Cost, right: Cost) -> Cost:
    """Return the exact non-negative difference ``left - right``."""

    left_coefficient, left_exponent = _parts(left, "left")
    right_coefficient, right_exponent = _parts(right, "right")
    if left < right:
        raise ValueError("cost subtraction cannot produce a negative result")
    if right_coefficient == 0:
        return left
    if left_coefficient == right_coefficient and left_exponent == right_exponent:
        return Decimal(0)
    common_exponent = min(left_exponent, right_exponent)
    coefficient = left_coefficient * 10 ** (left_exponent - common_exponent)
    coefficient -= right_coefficient * 10 ** (right_exponent - common_exponent)
    return _from_parts(coefficient, common_exponent)


def scale_cost(value: Cost, factor: int) -> Cost:
    """Multiply a cost by a non-negative integer without decimal rounding."""

    coefficient, exponent = _parts(value, "value")
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise TypeError("factor must be an integer")
    if factor < 0:
        raise ValueError("factor must be non-negative")
    return _from_parts(coefficient * factor, exponent)


def divide_cost(value: Cost, divisor: int) -> Cost:
    """Divide by a positive integer using an explicit deterministic contract.

    A terminating decimal quotient is constructed exactly. A repeating quotient has
    no exact ``Decimal`` representation, so it is rounded half-even with at least 50
    significant guard digits beyond the input coefficient. Ambient context settings
    never influence either path.
    """

    coefficient, exponent = _parts(value, "value")
    if isinstance(divisor, bool) or not isinstance(divisor, int):
        raise TypeError("divisor must be an integer")
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    if coefficient == 0 or divisor == 1:
        return value

    common_factor = gcd(coefficient, divisor)
    coefficient //= common_factor
    reduced_divisor = divisor // common_factor
    powers_of_two = 0
    powers_of_five = 0
    while reduced_divisor % 2 == 0:
        reduced_divisor //= 2
        powers_of_two += 1
    while reduced_divisor % 5 == 0:
        reduced_divisor //= 5
        powers_of_five += 1
    if reduced_divisor == 1:
        decimal_places = max(powers_of_two, powers_of_five)
        coefficient *= 2 ** (decimal_places - powers_of_two)
        coefficient *= 5 ** (decimal_places - powers_of_five)
        return _from_parts(coefficient, exponent - decimal_places)

    precision = max(50, len(value.as_tuple().digits) + 50)
    deterministic_context = Context(
        prec=precision,
        rounding=ROUND_HALF_EVEN,
        Emax=MAX_EMAX,
        Emin=MIN_EMIN,
    )
    with localcontext(deterministic_context):
        return value / Decimal(divisor)


def sum_costs(values: Iterable[Cost]) -> Cost:
    """Return an exact, context-independent sum of costs."""

    total = Decimal(0)
    for value in values:
        total = add_cost(total, value)
    return total
