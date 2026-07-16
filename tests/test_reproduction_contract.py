# SPDX-License-Identifier: Apache-2.0
"""Lock the public Make reproduction lanes to their documented scope."""

from pathlib import Path

MAKEFILE = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")


def _between(start: str, end: str) -> str:
    _, start_separator, after = MAKEFILE.partition(start)
    if not start_separator:
        raise AssertionError(f"Makefile reproduction boundary is missing: {start!r}")
    block, end_separator, _ = after.partition(end)
    if not end_separator:
        raise AssertionError(f"Makefile reproduction boundary is missing: {end!r}")
    return block


def test_reproduce_keeps_complete_backward_compatible_alias() -> None:
    assert "reproduce: reproduce-training\n" in MAKEFILE


def test_inference_reproduction_excludes_bilinear_and_lambda_training() -> None:
    recipe = _between(
        "reproduce-inference: install-dev\n",
        "\nreproduce-training: install-dev\n",
    )

    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in recipe
    assert "check-install" in recipe
    assert "scripts/smoke.py" in recipe
    assert "scripts/training_smoke.py" not in recipe
    assert "tierroute train" not in recipe
    assert "pytest" not in recipe
    assert "ruff" not in recipe
    assert " lint " not in recipe
    assert " test " not in recipe
    assert " licenses " not in recipe


def test_training_reproduction_preserves_complete_locked_pipeline() -> None:
    recipe = _between(
        "reproduce-training: install-dev\n",
        "\ndownload-routerbench:\n",
    )

    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in recipe
    for target in ("lint", "spdx", "test", "licenses", "check-install"):
        assert target in recipe
    assert "scripts/smoke.py" in recipe
    assert "scripts/training_smoke.py" in recipe
