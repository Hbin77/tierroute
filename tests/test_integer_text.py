# SPDX-License-Identifier: Apache-2.0
"""Tests for project-owned arbitrary-size integer encodings."""

from tierroute.core.integer_text import (
    decimal_to_integer,
    integer_identity_bytes,
    integer_to_decimal,
)
from tierroute.policies.resource_limits import MAX_POLICY_INTEGER_DECIMAL_DIGITS


def test_ten_thousand_digit_integer_round_trips_without_global_limit_changes() -> None:
    value = 10**10000 + 123456789

    document = integer_to_decimal(value)

    assert len(document) == 10001
    assert document.startswith("1")
    assert document.endswith("123456789")
    assert decimal_to_integer(document) == value
    assert decimal_to_integer(f"-{document}") == -value


def test_integer_identity_is_signed_and_self_delimiting() -> None:
    identities = {integer_identity_bytes(value) for value in (-256, -1, 0, 1, 256)}

    assert len(identities) == 5


def test_divide_and_conquer_leaf_boundaries_round_trip() -> None:
    for width in (599, 600, 601):
        document = "9" * width
        assert integer_to_decimal(decimal_to_integer(document)) == document


def test_maximum_policy_integer_width_round_trips() -> None:
    document = "9" * MAX_POLICY_INTEGER_DECIMAL_DIGITS

    assert integer_to_decimal(decimal_to_integer(document)) == document
