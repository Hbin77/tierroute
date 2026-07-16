# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the dependency-free raw GBM core."""

from collections.abc import Mapping, Sequence
from itertools import pairwise

import pytest

from tierroute.predictors.gbm import (
    GbmModel,
    GbmQualityPredictor,
    RegressionStump,
    fit_gradient_boosted_stumps,
)


def _fit(
    rows: Sequence[Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    *,
    ids: Sequence[str] | None = None,
    n_estimators: int = 2,
    learning_rate: float = 0.5,
    min_samples_leaf: int = 1,
    min_gain: float = 0.0,
) -> dict[str, GbmModel]:
    if ids is None:
        ids = tuple(f"e{index}" for index in range(len(rows)))
    return fit_gradient_boosted_stumps(
        rows,
        targets,
        example_ids=ids,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        min_gain=min_gain,
    )


def test_hand_computed_squared_error_boosting() -> None:
    model = _fit(
        ((0.0,), (1.0,), (2.0,), (3.0,)),
        {"model": (0.0, 0.0, 2.0, 2.0)},
    )["model"]

    assert model.base_value == pytest.approx(1.0)
    assert model.stumps == (
        RegressionStump(0, 2.0, -1.0, 1.0),
        RegressionStump(0, 2.0, -0.5, 0.5),
    )
    assert tuple(model.predict_features((value,)) for value in range(4)) == pytest.approx(
        (0.25, 0.25, 1.75, 1.75)
    )


def test_constant_target_early_stops_at_the_mean() -> None:
    model = _fit(
        ((0.0,), (1.0,), (2.0,), (3.0,)),
        {"model": (2.0, 2.0, 2.0, 2.0)},
        n_estimators=10,
    )["model"]

    assert model.base_value == 2.0
    assert model.stumps == ()
    assert model.predict_features((123.0,)) == 2.0


def test_tie_breaks_by_feature_and_uses_extreme_observed_right_value() -> None:
    model = _fit(
        (
            (-1.0e308, -1.0e308),
            (-1.0e308, -1.0e308),
            (1.0e308, 1.0e308),
            (1.0e308, 1.0e308),
        ),
        {"model": (0.0, 0.0, 2.0, 2.0)},
        n_estimators=1,
        learning_rate=1.0,
    )["model"]

    stump = model.stumps[0]
    assert stump.feature_index == 0
    assert stump.split_value == 1.0e308
    assert stump.predict_features((-1.0e308, 0.0)) == -1.0
    assert stump.predict_features((1.0e308, 0.0)) == 1.0


def test_equal_split_gains_choose_the_lower_observed_boundary() -> None:
    model = _fit(
        ((0.0,), (1.0,), (2.0,)),
        {"model": (0.0, 1.0, 0.0)},
        n_estimators=1,
        learning_rate=1.0,
    )["model"]

    assert model.stumps[0].split_value == 1.0


def test_per_model_heads_are_independent_and_unknown_model_fails() -> None:
    models = _fit(
        ((0.0,), (1.0,), (2.0,), (3.0,)),
        {
            "cheap": (0.0, 0.0, 2.0, 2.0),
            "premium": (4.0, 4.0, 0.0, 0.0),
        },
        n_estimators=1,
        learning_rate=1.0,
    )
    predictor = GbmQualityPredictor(
        vectorizer=lambda prompt: (float(prompt),),
        models=models,
    )

    assert predictor.predict_many("0", ("cheap", "premium")) == pytest.approx(
        {"cheap": 0.0, "premium": 4.0}
    )
    with pytest.raises(KeyError, match="no GBM head"):
        predictor.predict("0", "missing")


def test_predict_batch_calls_batch_vectorizer_once() -> None:
    calls: list[tuple[str, ...]] = []
    model = GbmModel(
        feature_width=1,
        base_value=1.0,
        learning_rate=0.5,
        stumps=(RegressionStump(0, 1.0, -1.0, 1.0),),
    )

    def vectorize_batch(prompts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        calls.append(tuple(prompts))
        return tuple((float(prompt),) for prompt in prompts)

    predictor = GbmQualityPredictor(
        vectorizer=lambda _: pytest.fail("scalar vectorizer must not be called"),
        models={"model": model},
        batch_vectorizer=vectorize_batch,
    )

    assert predictor.predict_batch(("0", "2"), ("model",)) == (
        {"model": 0.5},
        {"model": 1.5},
    )
    assert calls == [("0", "2")]


def test_training_order_is_canonicalized_by_example_id() -> None:
    rows = ((0.0, 1.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0))
    targets = {"z": (0.0, 0.5, 2.0, 2.5), "a": (3.0, 2.0, 1.0, 0.0)}
    expected = _fit(rows, targets, ids=("a", "b", "c", "d"), n_estimators=4)
    permutation = (2, 0, 3, 1)

    actual = _fit(
        tuple(rows[index] for index in permutation),
        {
            model_id: tuple(values[index] for index in permutation)
            for model_id, values in reversed(tuple(targets.items()))
        },
        ids=tuple(("a", "b", "c", "d")[index] for index in permutation),
        n_estimators=4,
    )

    assert tuple(actual) == ("a", "z")
    assert actual == expected


def test_every_stored_boosting_round_strictly_reduces_training_loss() -> None:
    rows = ((0.0,), (1.0,), (2.0,), (3.0,), (4.0,), (5.0,))
    targets = (0.0, 1.0, 1.0, 3.0, 5.0, 8.0)
    model = _fit(
        rows,
        {"model": targets},
        n_estimators=8,
        learning_rate=0.25,
    )["model"]
    predictions = [model.base_value] * len(rows)
    losses = [
        sum(
            (target - prediction) ** 2
            for target, prediction in zip(targets, predictions, strict=True)
        )
    ]

    for stump in model.stumps:
        predictions = [
            prediction + model.learning_rate * stump.predict_features(row)
            for prediction, row in zip(predictions, rows, strict=True)
        ]
        losses.append(
            sum(
                (target - prediction) ** 2
                for target, prediction in zip(targets, predictions, strict=True)
            )
        )

    assert len(model.stumps) == 8
    assert all(after < before for before, after in pairwise(losses))


def test_fitter_rejects_non_finite_ragged_or_misaligned_data() -> None:
    with pytest.raises(ValueError, match="consistent width"):
        _fit(((0.0,), (1.0, 2.0)), {"model": (0.0, 1.0)})
    with pytest.raises(ValueError, match="must be finite"):
        _fit(((0.0,), (float("nan"),)), {"model": (0.0, 1.0)})
    with pytest.raises(ValueError, match="target count"):
        _fit(((0.0,), (1.0,)), {"model": (0.0,)})
    with pytest.raises(ValueError, match="target model IDs"):
        _fit(((0.0,), (1.0,)), {"": (0.0, 1.0)})


def test_models_and_prediction_rows_must_share_a_valid_shape() -> None:
    one_wide = GbmModel(feature_width=1, base_value=0.0, learning_rate=0.1)
    two_wide = GbmModel(feature_width=2, base_value=0.0, learning_rate=0.1)

    with pytest.raises(ValueError, match="same feature width"):
        GbmQualityPredictor(lambda _: (0.0,), {"one": one_wide, "two": two_wide})
    predictor = GbmQualityPredictor(lambda _: (0.0, 1.0), {"one": one_wide})
    with pytest.raises(ValueError, match="does not match expected width"):
        predictor.predict("prompt", "one")
