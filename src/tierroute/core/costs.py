# SPDX-License-Identifier: Apache-2.0
"""Context-independent arithmetic for exact decimal costs.

``Decimal`` operators obey the caller's mutable global context. That is useful for
general numerical work, but unsafe at a budget boundary: a low-precision context can
round a small overspend away. These helpers operate on decimal coefficient/exponent
tuples and Python integers, so finite additions, subtractions, and integer scaling do
not round and do not depend on ambient process state. The explicit range in
``core.schemas`` bounds coefficient/exponent expansion and applies to every result.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, Context, Decimal, localcontext
from math import gcd

from tierroute.core.integer_text import decimal_to_integer, integer_to_decimal
from tierroute.core.schemas import (
    MAX_COST_DECIMAL_DIGITS,
    Cost,
    _require_non_negative_finite_cost,
)

_MAX_COST_COEFFICIENT_EXCLUSIVE = 10**MAX_COST_DECIMAL_DIGITS
_MAX_COST_SCALE_FACTOR_EXCLUSIVE = 10 ** (2 * MAX_COST_DECIMAL_DIGITS)


def _parts(value: Cost, field_name: str) -> tuple[int, int, int]:
    _require_non_negative_finite_cost(value, field_name)
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"{field_name} must be finite")
    trailing_zeros = 0
    for digit in reversed(digits):
        if digit != 0:
            break
        trailing_zeros += 1
    if trailing_zeros == len(digits):
        return 0, 0, 1
    significant = digits[: len(digits) - trailing_zeros]
    coefficient = decimal_to_integer("".join(chr(ord("0") + digit) for digit in significant))
    exponent += trailing_zeros
    if sign:
        coefficient = -coefficient
    return coefficient, exponent, len(significant)


def _factor_power(value: int, base: int) -> tuple[int, int]:
    """Return the exact base-adic order and remaining positive cofactor."""

    if value % base:
        return 0, value
    if base == 2:
        exponent = (value & -value).bit_length() - 1
        return exponent, value >> exponent
    # For base five, bit_length // 2 + 1 is a safe upper bound because 5 > 2**2.
    low = 1
    high = value.bit_length() // 2 + 1
    while low < high:
        middle = (low + high + 1) // 2
        if value % (base**middle) == 0:
            low = middle
        else:
            high = middle - 1
    factor = base**low
    return low, value // factor


def _strip_decimal_zeros(value: int) -> tuple[int, int]:
    powers_of_two, _ = _factor_power(value, 2)
    powers_of_five, _ = _factor_power(value, 5)
    decimal_zeros = min(powers_of_two, powers_of_five)
    if decimal_zeros:
        value //= 10**decimal_zeros
    return value, decimal_zeros


def _strip_product_decimal_zeros(left: int, right: int) -> tuple[int, int, int]:
    """Remove every factor of ten formed within or across two positive operands."""

    left_twos, _ = _factor_power(left, 2)
    left_fives, _ = _factor_power(left, 5)
    right_twos, _ = _factor_power(right, 2)
    right_fives, _ = _factor_power(right, 5)
    decimal_zeros = min(left_twos + right_twos, left_fives + right_fives)

    left_twos_to_remove = min(left_twos, decimal_zeros)
    right_twos_to_remove = decimal_zeros - left_twos_to_remove
    left_fives_to_remove = min(left_fives, decimal_zeros)
    right_fives_to_remove = decimal_zeros - left_fives_to_remove
    left >>= left_twos_to_remove
    right >>= right_twos_to_remove
    if left_fives_to_remove:
        left //= 5**left_fives_to_remove
    if right_fives_to_remove:
        right //= 5**right_fives_to_remove
    return left, right, decimal_zeros


def _from_parts(coefficient: int, exponent: int) -> Cost:
    if coefficient == 0:
        return Decimal(0)
    sign = int(coefficient < 0)
    coefficient = abs(coefficient)

    # A canonical result avoids retaining redundant zero digits after repeated sums.
    coefficient, decimal_zeros = _strip_decimal_zeros(coefficient)
    exponent += decimal_zeros
    if coefficient >= _MAX_COST_COEFFICIENT_EXCLUSIVE:
        raise ValueError("cost result exceeds the supported exact decimal digit range")
    document = integer_to_decimal(coefficient)
    result = Decimal((sign, tuple(ord(character) - ord("0") for character in document), exponent))
    _require_non_negative_finite_cost(result, "cost result")
    return result


def add_cost(left: Cost, right: Cost) -> Cost:
    """Return the exact sum of two non-negative finite costs."""

    left_coefficient, left_exponent, left_digits = _parts(left, "left")
    right_coefficient, right_exponent, right_digits = _parts(right, "right")
    if left_coefficient == 0:
        return right
    if right_coefficient == 0:
        return left
    common_exponent = min(left_exponent, right_exponent)
    highest_exclusive = max(
        left_exponent + left_digits,
        right_exponent + right_digits,
    )
    if highest_exclusive - common_exponent > MAX_COST_DECIMAL_DIGITS:
        raise ValueError("cost addition exceeds the supported exact decimal digit range")
    coefficient = left_coefficient * 10 ** (left_exponent - common_exponent)
    coefficient += right_coefficient * 10 ** (right_exponent - common_exponent)
    return _from_parts(coefficient, common_exponent)


def subtract_cost(left: Cost, right: Cost) -> Cost:
    """Return the exact non-negative difference ``left - right``."""

    left_coefficient, left_exponent, _ = _parts(left, "left")
    right_coefficient, right_exponent, _ = _parts(right, "right")
    if left < right:
        raise ValueError("cost subtraction cannot produce a negative result")
    if right_coefficient == 0:
        return left
    if left_coefficient == right_coefficient and left_exponent == right_exponent:
        return Decimal(0)
    common_exponent = min(left_exponent, right_exponent)
    # Legal operands bound this temporary alignment to fewer than twice the public
    # range. Validate only after subtraction: cancellation can turn that wider
    # intermediate into a legal one-digit result at the minimum exponent.
    coefficient = left_coefficient * 10 ** (left_exponent - common_exponent)
    coefficient -= right_coefficient * 10 ** (right_exponent - common_exponent)
    return _from_parts(coefficient, common_exponent)


def scale_cost(value: Cost, factor: int) -> Cost:
    """Multiply by an integer after removing internal and cross decimal powers."""

    coefficient, exponent, _ = _parts(value, "value")
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise TypeError("factor must be an integer")
    if factor < 0:
        raise ValueError("factor must be non-negative")
    if coefficient == 0 or factor == 0:
        return Decimal(0)
    # Any legal nonzero cost is at least 1e-N and every legal result is below 1eN,
    # so a factor at or above 1e(2N) cannot possibly fit. This cheap numeric guard
    # bounds the valuation work without rejecting cross-factor cancellation.
    if factor >= _MAX_COST_SCALE_FACTOR_EXCLUSIVE:
        raise ValueError("factor exceeds the supported exact decimal digit range")
    coefficient, factor, decimal_zeros = _strip_product_decimal_zeros(coefficient, factor)
    exponent += decimal_zeros
    if factor >= _MAX_COST_COEFFICIENT_EXCLUSIVE:
        raise ValueError("factor exceeds the supported exact decimal digit range")
    return _from_parts(coefficient * factor, exponent)


def divide_cost(value: Cost, divisor: int) -> Cost:
    """Divide by a positive integer using an explicit deterministic contract.

    A terminating decimal quotient is constructed exactly. A repeating quotient has
    no exact ``Decimal`` representation, so it is rounded half-even with at least 50
    significant guard digits beyond the reduced exact quotient numerator. Common
    factors are removed before decimal powers in the divisor are absorbed into the
    exponent and the remaining cofactor is bounded. Ambient context settings never
    influence either path.
    """

    coefficient, exponent, _ = _parts(value, "value")
    if isinstance(divisor, bool) or not isinstance(divisor, int):
        raise TypeError("divisor must be an integer")
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    if coefficient == 0:
        return value
    # Reduce before bounding the divisor: a very wide caller representation can
    # cancel completely against a legal coefficient (for example c / (2*c)).
    common_factor = gcd(coefficient, divisor)
    coefficient //= common_factor
    divisor //= common_factor
    divisor, decimal_zeros = _strip_decimal_zeros(divisor)
    exponent -= decimal_zeros
    if divisor >= _MAX_COST_COEFFICIENT_EXCLUSIVE:
        raise ValueError("divisor exceeds the supported exact decimal digit range")
    if divisor == 1:
        return _from_parts(coefficient, exponent)

    canonical_value = _from_parts(coefficient, exponent)
    significant_digits = len(integer_to_decimal(coefficient))
    reduced_divisor = divisor
    powers_of_two, reduced_divisor = _factor_power(reduced_divisor, 2)
    powers_of_five, reduced_divisor = _factor_power(reduced_divisor, 5)
    if reduced_divisor == 1:
        decimal_places = max(powers_of_two, powers_of_five)
        if exponent - decimal_places < -MAX_COST_DECIMAL_DIGITS:
            raise ValueError("terminating cost quotient exceeds the supported exact cost range")
        coefficient *= 2 ** (decimal_places - powers_of_two)
        coefficient *= 5 ** (decimal_places - powers_of_five)
        return _from_parts(coefficient, exponent - decimal_places)

    precision = max(50, significant_digits + 50)
    if precision > MAX_COST_DECIMAL_DIGITS:
        raise ValueError("repeating cost quotient exceeds the supported precision range")
    deterministic_context = Context(
        prec=precision,
        rounding=ROUND_HALF_EVEN,
        Emax=MAX_EMAX,
        Emin=MIN_EMIN,
        traps=[],
    )
    with localcontext(deterministic_context):
        result = canonical_value / Decimal(divisor)
    _require_non_negative_finite_cost(result, "cost quotient")
    return result


def sum_costs(values: Iterable[Cost]) -> Cost:
    """Return an exact, context-independent sum of costs."""

    total = Decimal(0)
    for value in values:
        total = add_cost(total, value)
    return total
