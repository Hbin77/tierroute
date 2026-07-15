# SPDX-License-Identifier: Apache-2.0
"""Shared resource contracts for exact-lambda policy artifacts."""

from tierroute.core.schemas import MAX_COST_DECIMAL_DIGITS

MAX_POLICY_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_POLICY_LEDGER_ADAPTER_NAME_BYTES = 4 * 1024
# A root divides an exact binary64 quality difference by a legal Decimal cost
# difference; an adjacent midpoint can multiply two such denominators. Four cost
# spans plus binary64 and carry headroom cover every automatically derived candidate.
MAX_POLICY_INTEGER_DECIMAL_DIGITS = 4 * MAX_COST_DECIMAL_DIGITS + 4_096
MAX_POLICY_CANDIDATES_PER_TIER = 100_000
