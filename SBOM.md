<!-- SPDX-License-Identifier: Apache-2.0 -->

# Software Bill of Materials

Last audited: 2026-07-16. Update this file in the same commit whenever a dependency,
model, dataset, font, media asset, or CI action is added or upgraded.

The core `tierroute` runtime has no third-party Python dependency. The pinned
RouterBench artifact is decoded with the Python standard library; development tools are
isolated from the offline routing path. Predictor training uses the project-owned
standard-library centered-ridge solver and adds no distribution dependency.

## Build dependency

| Component | Version | License | Source | Purpose | Distribution |
|---|---:|---|---|---|---|
| flit_core | 3.12.0 | BSD-3-Clause | https://github.com/pypa/flit | PEP 517/660 build backend | Build and editable-install time only |
| Tomli (vendored by flit_core) | 1.2.3 | MIT | https://github.com/hukkin/tomli | Parse `pyproject.toml` on older Python versions | Bundled build-time component; not a separate distribution |
| packaging version regex (vendored by flit_core) | flit_core 3.12.0 snapshot | BSD-2-Clause | https://github.com/pypa/packaging | Normalize PEP 440 versions | Bundled build-time code fragment; not a separate distribution |
| SPDX License List identifiers (vendored by flit_core) | flit_core 3.12.0 snapshot | CC0-1.0 | https://github.com/spdx/license-list-data | Validate PEP 639 license expressions | Generated build-time data; not a separate distribution |

## Development and CI dependencies

Exact versions are recorded in `requirements-dev.lock`.

| Component | Version | License | Source | Purpose | Relationship |
|---|---:|---|---|---|---|
| pytest | 8.4.2 | MIT | https://github.com/pytest-dev/pytest | Test runner | Direct |
| ruff | 0.15.21 | MIT | https://github.com/astral-sh/ruff | Lint and format checks | Direct |
| pip-licenses | 5.5.5 | MIT | https://github.com/raimon49/pip-licenses | CI license gate | Direct |
| flit_core | 3.12.0 | BSD-3-Clause | https://github.com/pypa/flit | Locked no-build-isolation editable installs | Direct build tool |
| iniconfig | 2.3.0 | MIT | https://github.com/pytest-dev/iniconfig | pytest configuration | Transitive |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging | Version/specifier handling | Transitive |
| pip | 26.1.2 | MIT | https://github.com/pypa/pip | Deterministic environment installer and wheel frontend | Direct development/CI tool; not shipped at runtime |
| pluggy | 1.6.0 | MIT | https://github.com/pytest-dev/pluggy | pytest plugin system | Transitive |
| Pygments | 2.20.0 | BSD-2-Clause | https://github.com/pygments/pygments | pytest trace highlighting | Transitive |
| prettytable | 3.18.0 | BSD-3-Clause | https://github.com/prettytable/prettytable | pip-licenses output | Transitive |
| wcwidth | 0.8.2 | MIT | https://github.com/jquast/wcwidth | prettytable terminal widths | Transitive |
| exceptiongroup | 1.3.1 | MIT | https://github.com/agronholm/exceptiongroup | pytest exception groups on Python 3.10 | Conditional transitive (`python_version < 3.11`) |
| tomli | 2.4.1 | MIT | https://github.com/hukkin/tomli | TOML parsing on Python 3.10 | Conditional transitive (`python_version < 3.11`) |
| typing-extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions | Backported typing APIs on Python 3.10 | Conditional transitive (`python_version < 3.11`) |

## Models and model-serving assets

| Asset | Revision | License | Source | Purpose | Repository status |
|---|---|---|---|---|---|
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | MIT | https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181 | Planned multilingual prompt embeddings | Not downloaded or distributed in W1; runtime contract accepts local paths only |

No LLM weights or commercial API client is bundled. The default CLI predictor and the
optional fitted bilinear artifact use project-authored logic; neither is an LLM nor a
benchmark result.

## Data assets

| Asset | Revision / checksum | License | Source | Purpose | Repository status |
|---|---|---|---|---|---|
| tierroute synthetic smoke dataset | `src/tierroute/data/synthetic.json` | Apache-2.0 | Project-authored | Clone-without-download quickstart and CI | Distributed; sidecar SPDX license included |
| RouterBench 0-shot | HF revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`; artifact SHA-256 `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`; decoded semantic SHA-256 `7b4749ad5c4bdb338c2317b306c382680b1a23dc83c73e29ab805b8f7e472e87` | NOASSERTION | https://huggingface.co/datasets/withmartian/routerbench/tree/784021482c3f320c6619ed4b3bb3b41a21424fcb | Optional external harness validation | Never committed or redistributed; upstream license clarification required |

RouterBench's GitHub code repository is MIT-licensed, but that declaration does not
license the separate Hugging Face dataset. tierroute contains no copied RouterBench code.

## CI actions

| Component | Pinned commit | License | Source | Purpose |
|---|---|---|---|---|
| actions/checkout | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` (v7.0.0) | MIT | https://github.com/actions/checkout | Read repository contents in CI |
| actions/setup-python | `ece7cb06caefa5fff74198d8649806c4678c61a1` (v6.3.0) | MIT | https://github.com/actions/setup-python | Install matrix Python versions |

## License gate

CI installs the exact development lock, scans top-level distribution metadata with
`pip-licenses`, and separately inspects installed license documents plus nested
vendored `.dist-info/METADATA`. Either layer rejects GPL, LGPL, or AGPL family terms;
top-level metadata must also match the reviewed allowlist. CI installs the base wheel
into a fresh environment and asserts that flit_core, setuptools, pandas, and NumPy are
absent. Document-level exceptions are exact reviewed PSF-family license evidence for
`typing_extensions==4.16.0` and pip's vendored `distlib==0.4.0`; their hashes and audit
trail are recorded in `docs/dependency-license-audit.md`, and modified evidence is
scanned normally. This automated scan still does not replace pre-adoption review of
native binary linkage, models, datasets, vendored files, or GitHub Actions. Those
assets must also be added to this SBOM before adoption.
