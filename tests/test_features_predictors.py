# SPDX-License-Identifier: Apache-2.0
"""Tests for offline prompt features and quality prediction primitives."""

import pytest

from tierroute.features import LocalEmbeddingModel, extract_surface_features
from tierroute.predictors import (
    BilinearQualityPredictor,
    CalibratedQualityPredictor,
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
    StaticQualityPredictor,
)


def test_surface_features_detect_korean_math_and_code() -> None:
    prompt = "파이썬으로 방정식을 풀어줘\n```python\ndef solve(): pass\n```"
    features = extract_surface_features(prompt)

    assert features.line_count == 4
    assert features.has_code is True
    assert features.has_math is True
    assert set(features.domain_tags) >= {"code", "math"}


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("render <a>link</a>", True),
        ("render </A data-value='x\ny'>", True),
        ("plain\rdef example(): pass", True),
        ("compare <1 and >2", False),
        ("<a" * 100_000, False),
        ("\n " * 100_000 + "x", False),
    ],
)
def test_surface_html_code_signal_is_linear_and_compatible(
    prompt: str,
    expected: bool,
) -> None:
    assert extract_surface_features(prompt).has_code is expected


def test_local_embedding_model_never_falls_back_to_network(tmp_path: object) -> None:
    model = LocalEmbeddingModel(tmp_path / "missing")  # type: ignore[operator]

    with pytest.raises(FileNotFoundError, match="prepare it explicitly"):
        model.validate()


def test_bilinear_predictor_checks_vector_width() -> None:
    predictor = BilinearQualityPredictor(
        vectorizer=lambda _: (2.0, 3.0),
        model_weights={"small": (0.5, 1.0)},
        model_bias={"small": 0.25},
    )

    assert predictor.predict("prompt", "small") == pytest.approx(4.25)


def test_isotonic_calibrator_merges_decreasing_blocks() -> None:
    calibrator = IsotonicCalibrator.fit([0.1, 0.2, 0.3], [0.2, 0.8, 0.4])

    assert calibrator.values == pytest.approx((0.2, 0.6))
    assert calibrator.calibrate(0.25) == pytest.approx(0.6)
    wrapped = CalibratedQualityPredictor(StaticQualityPredictor({"m": 0.25}), calibrator)
    assert wrapped.predict("prompt", "m") == pytest.approx(0.6)


def test_per_model_calibration_rejects_malformed_batch_output() -> None:
    class BrokenBatchPredictor:
        def predict(self, prompt: str, model_id: str) -> float:
            del prompt, model_id
            return 0.5

        def predict_many(self, prompt: str, model_ids: object) -> dict[str, float]:
            del prompt, model_ids
            return {"unexpected": 0.5}

        def predict_batch(self, prompts: object, model_ids: object) -> tuple[dict[str, float], ...]:
            del prompts, model_ids
            return ({"unexpected": 0.5},)

    calibrator = IsotonicCalibrator((0.5,), (0.5,))
    predictor = PerModelCalibratedQualityPredictor(
        BrokenBatchPredictor(),
        {"m": calibrator},
    )

    with pytest.raises(ValueError, match="every requested model"):
        predictor.predict_many("prompt", ("m",))
    with pytest.raises(ValueError, match="every requested model"):
        predictor.predict_batch(("prompt",), ("m",))
