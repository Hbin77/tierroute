# SPDX-License-Identifier: Apache-2.0

PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help install install-dev check-install lint test licenses spdx smoke training-smoke
.PHONY: verify reproduce
.PHONY: download-routerbench validate-routerbench

help:
	@echo "tierroute developer targets"
	@echo "  install              install tierroute in editable mode"
	@echo "  install-dev          install the exact dev lock and editable tierroute"
	@echo "  verify               run static checks, tests, licenses, and offline smoke"
	@echo "  reproduce            install and run the complete no-external-data pipeline"
	@echo "  training-smoke       fit, load, and route with a local predictor artifact"
	@echo "  download-routerbench explicitly download pinned RouterBench data (network)"
	@echo "  validate-routerbench validate a previously downloaded local artifact"

install:
	$(PYTHON) -m pip install --no-deps -e .

install-dev:
	$(PYTHON) -m pip install --no-deps --requirement requirements-dev.lock
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

check-install:
	$(PYTHON) -m pip check
	$(PYTHON) -c "import importlib.metadata as m; assert m.version('tierroute')"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest

licenses:
	$(PYTHON) scripts/check_licenses.py

spdx:
	$(PYTHON) scripts/check_spdx.py

smoke:
	@set -eu; \
	hf_home="$$(mktemp -d)"; \
	python_bin="$$($(PYTHON) -c 'import os, sys; print(os.path.dirname(sys.executable))')"; \
	trap 'rm -rf "$$hf_home"' EXIT HUP INT TERM; \
	PATH="$$python_bin:$$PATH" HF_HOME="$$hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		$(PYTHON) scripts/smoke.py

training-smoke:
	@set -eu; \
	hf_home="$$(mktemp -d)"; \
	python_bin="$$($(PYTHON) -c 'import os, sys; print(os.path.dirname(sys.executable))')"; \
	trap 'rm -rf "$$hf_home"' EXIT HUP INT TERM; \
	PATH="$$python_bin:$$PATH" HF_HOME="$$hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		$(PYTHON) scripts/training_smoke.py

verify: lint spdx test licenses check-install smoke training-smoke

reproduce: install-dev
	@set -eu; \
	hf_home="$$(mktemp -d)"; \
	python_bin="$$($(PYTHON) -c 'import os, sys; print(os.path.dirname(sys.executable))')"; \
	trap 'rm -rf "$$hf_home"' EXIT HUP INT TERM; \
	export PATH="$$python_bin:$$PATH"; \
	export HF_HOME="$$hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
	$(MAKE) --no-print-directory lint spdx test licenses check-install PYTHON="$(PYTHON)"; \
	$(PYTHON) scripts/smoke.py; \
	$(PYTHON) scripts/training_smoke.py

download-routerbench:
	@echo "Network access: downloading the explicitly opted-in RouterBench artifact."
	$(PYTHON) scripts/download_routerbench.py

validate-routerbench:
	$(PYTHON) scripts/validate_routerbench.py
