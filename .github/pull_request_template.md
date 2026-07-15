<!-- SPDX-License-Identifier: Apache-2.0 -->

## Summary

<!-- What problem does this solve, and why is this the right package boundary? -->

## AI assistance disclosure

<!-- Use `None` when no AI assistant materially contributed. Do not guess a model snapshot or line percentage. -->

- Tool/service used:
- Exact model/version/snapshot, or `not exposed/not retained`:
- Material activities (requirements/design/code/tests/docs/debugging/review):
- Affected paths or algorithms:
- Human validation performed in this PR (or `Pending`/`None`):
- Audit-ledger entry (see `docs/ai-assistance-audit.md`):

## Verification

<!-- List exact commands and relevant results. Do not claim synthetic scores as benchmarks. -->

```text
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 pytest
tierroute route "offline smoke" --tier fast
tierroute evaluate
tierroute demo
```

## Checklist

- [ ] The change is focused and linked to an issue, or the motivation is explained above.
- [ ] Runtime routing, feature extraction, prediction, and evaluation remain network-free.
- [ ] New or changed project-authored files include an Apache-2.0 SPDX identifier where comments are supported.
- [ ] Tests cover the behavior and failure path; the commands above pass.
- [ ] Modeling evaluation uses LODO with no held-out-domain fitting or other label leakage.
- [ ] No private/unlicensed data, model weights, credentials, generated secrets, or arbitrary pickle artifacts are committed.
- [ ] Every third-party asset and direct/transitive dependency has a compatible, documented license; no GPL-family dependency is introduced.
- [ ] `SBOM.md` and user documentation are updated for dependency, asset, interface, or reproducibility changes.
- [ ] Challenge-specific uncertainty stays in `adapters/`; cascade behavior is not enabled without confirmed sequential-call semantics.
- [ ] Material AI assistance is disclosed without an invented model snapshot or authored-line percentage.
- [ ] AI-agent, subagent, and CI review is not represented as independent human review or owner sign-off.
- [ ] A contest-critical change links its named human walkthrough record, or explicitly records that walkthrough as pending.
- [ ] Commits follow Conventional Commits and contain no unrelated generated-file churn.
