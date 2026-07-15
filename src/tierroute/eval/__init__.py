# SPDX-License-Identifier: Apache-2.0
"""Offline evaluation and budget simulation."""

from tierroute.eval.budgets import BudgetLedger, BudgetLedgerFactory
from tierroute.eval.metrics import (
    ScoreSummary,
    oracle_gap_recovery,
    summarize_report,
    weighted_delta,
)
from tierroute.eval.planning import (
    DomainTablePlan,
    build_per_query_oracle_plan,
    fit_per_query_domain_table,
)
from tierroute.eval.schemas import (
    BudgetReport,
    CandidateOutcome,
    EvaluationExample,
    EvaluationReport,
    QueryResult,
    TierResult,
    TierSpec,
)
from tierroute.eval.simulator import OfflineSimulator
from tierroute.eval.validation import DomainFold, leave_one_domain_out

__all__ = [
    "BudgetLedger",
    "BudgetLedgerFactory",
    "BudgetReport",
    "CandidateOutcome",
    "DomainFold",
    "DomainTablePlan",
    "EvaluationExample",
    "EvaluationReport",
    "OfflineSimulator",
    "QueryResult",
    "ScoreSummary",
    "TierResult",
    "TierSpec",
    "build_per_query_oracle_plan",
    "fit_per_query_domain_table",
    "leave_one_domain_out",
    "oracle_gap_recovery",
    "summarize_report",
    "weighted_delta",
]
