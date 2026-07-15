<!-- SPDX-License-Identifier: Apache-2.0 -->

# Software Bill of Materials

Last audited: 2026-07-15. Update this file in the same commit whenever a dependency,
model, dataset, font, media asset, or CI action is added or upgraded.

The core `tierroute` runtime has no third-party Python dependency. Optional data readers
and development tools are isolated from the offline routing path.

## Build dependency

| Component | Version | License | Source | Purpose | Distribution |
|---|---:|---|---|---|---|
| setuptools | 83.0.0 | MIT | https://github.com/pypa/setuptools | PEP 517 build backend | Build time only |

## Development and CI dependencies

Exact versions are recorded in `requirements-dev.lock`.

| Component | Version | License | Source | Purpose | Relationship |
|---|---:|---|---|---|---|
| pytest | 8.4.2 | MIT | https://github.com/pytest-dev/pytest | Test runner | Direct |
| ruff | 0.15.21 | MIT | https://github.com/astral-sh/ruff | Lint and format checks | Direct |
| pip-licenses | 5.5.5 | MIT | https://github.com/raimon49/pip-licenses | CI license gate | Direct |
| iniconfig | 2.3.0 | MIT | https://github.com/pytest-dev/iniconfig | pytest configuration | Transitive |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging | Version/specifier handling | Transitive |
| pluggy | 1.6.0 | MIT | https://github.com/pytest-dev/pluggy | pytest plugin system | Transitive |
| Pygments | 2.20.0 | BSD-2-Clause | https://github.com/pygments/pygments | pytest trace highlighting | Transitive |
| prettytable | 3.18.0 | BSD-3-Clause | https://github.com/prettytable/prettytable | pip-licenses output | Transitive |
| wcwidth | 0.8.2 | MIT | https://github.com/jquast/wcwidth | prettytable terminal widths | Transitive |
| exceptiongroup | 1.3.1 | MIT | https://github.com/agronholm/exceptiongroup | pytest exception groups on Python 3.10 | Conditional transitive (`python_version < 3.11`) |
| tomli | 2.4.1 | MIT | https://github.com/hukkin/tomli | TOML parsing on Python 3.10 | Conditional transitive (`python_version < 3.11`) |
| typing-extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions | Backported typing APIs on Python 3.10 | Conditional transitive (`python_version < 3.11`) |

## Optional RouterBench reader

These packages are not installed by `pip install -e .`. They are pinned in
`requirements-routerbench.lock` and the `routerbench` extra solely to authenticate and
read the upstream pickle during opt-in local validation.

| Component | Version | License | Source | Purpose | Relationship |
|---|---:|---|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause | https://github.com/numpy/numpy | pandas numerical arrays | Direct optional pin |
| pandas | 2.3.3 | BSD-3-Clause | https://github.com/pandas-dev/pandas | Read authenticated RouterBench DataFrame pickle | Direct optional pin |
| python-dateutil | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause | https://github.com/dateutil/dateutil | pandas date utilities | Transitive |
| pytz | 2026.2 | MIT | https://github.com/stub42/pytz | pandas timezone data | Transitive |
| six | 1.17.0 | MIT | https://github.com/benjaminp/six | python-dateutil compatibility | Transitive |
| tzdata | 2026.3 | Apache-2.0 | https://github.com/python/tzdata | pandas timezone database | Transitive |

## Models and model-serving assets

| Asset | Revision | License | Source | Purpose | Repository status |
|---|---|---|---|---|---|
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | MIT | https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181 | Planned multilingual prompt embeddings | Not downloaded or distributed in W1; runtime contract accepts local paths only |

No LLM weights or commercial API client is bundled. The CLI's quality predictor is
project-authored deterministic demo logic, not an AI model or benchmark result.

## Data assets

| Asset | Revision / checksum | License | Source | Purpose | Repository status |
|---|---|---|---|---|---|
| tierroute synthetic smoke dataset | `src/tierroute/data/synthetic.json` | Apache-2.0 | Project-authored | Clone-without-download quickstart and CI | Distributed; sidecar SPDX license included |
| RouterBench 0-shot | HF revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`; SHA-256 `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d` | NOASSERTION | https://huggingface.co/datasets/withmartian/routerbench | Optional external harness validation | Never committed or redistributed; upstream license clarification required |

RouterBench's GitHub code repository is MIT-licensed, but that declaration does not
license the separate Hugging Face dataset. tierroute contains no copied RouterBench code.

## CI actions

| Component | Pinned commit | License | Source | Purpose |
|---|---|---|---|---|
| actions/checkout | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` (v7.0.0) | MIT | https://github.com/actions/checkout | Read repository contents in CI |
| actions/setup-python | `ece7cb06caefa5fff74198d8649806c4678c61a1` (v6.3.0) | MIT | https://github.com/actions/setup-python | Install matrix Python versions |

## License gate

CI installs both the exact development lock and the optional RouterBench reader lock,
then scans that clean environment with `pip-licenses`. It rejects GPL, LGPL, and AGPL
family metadata and permits only the reviewed permissive license set. This scan does
not replace manual review of models, datasets, vendored files, or GitHub Actions; those
assets must also be added to this SBOM before adoption.
