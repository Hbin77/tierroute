# SPDX-License-Identifier: Apache-2.0
"""Nominal marker for the evaluation-only privileged replay boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tierroute.core import RouterAction, RouterState


class PrivilegedEvaluationRouter(ABC):
    """A non-deployable upper-bound router with access to a hidden example key.

    The key is passed out-of-band by :class:`OfflineSimulator`; it is never added
    to :class:`RouterState`, so ordinary policies cannot infer a held-out domain
    from dataset-specific identifiers.
    """

    @abstractmethod
    def route_with_evaluation_context(self, state: RouterState, *, example_id: str) -> RouterAction:
        """Choose an action using a private evaluation identifier."""
