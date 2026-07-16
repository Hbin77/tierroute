# SPDX-License-Identifier: Apache-2.0
"""Model-quality prediction, training, calibration, and artifacts."""

from tierroute.predictors.artifacts import (
    PREDICTOR_ARTIFACT_VERSION,
    BilinearPredictorArtifact,
)
from tierroute.predictors.base import (
    BatchPromptQualityPredictor,
    BatchQualityPredictor,
    BilinearQualityPredictor,
    QualityPredictor,
    StaticQualityPredictor,
)
from tierroute.predictors.calibration import (
    CalibratedQualityPredictor,
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
)
from tierroute.predictors.gbm import GbmModel, GbmQualityPredictor, RegressionStump
from tierroute.predictors.gbm_training import (
    GBM_ALGORITHM_ID,
    GbmNestedLodoWorkEstimate,
    GbmTrainingConfig,
    estimate_nested_lodo_gbm_work,
    fit_calibrated_gbm,
    fit_calibrated_gbm_for_fold,
    preflight_gbm_fit,
    preflight_nested_lodo_gbm,
)
from tierroute.predictors.solvers import (
    KNOWN_RIDGE_SOLVER_IDS,
    NATIVE_C11_RIDGE_SOLVER_ID,
    RidgeSolver,
    resolve_ridge_solver,
)
from tierroute.predictors.training import (
    BilinearTrainingConfig,
    fit_calibrated_bilinear,
    fit_calibrated_bilinear_for_fold,
    training_data_sha256,
)

__all__ = [
    "GBM_ALGORITHM_ID",
    "KNOWN_RIDGE_SOLVER_IDS",
    "NATIVE_C11_RIDGE_SOLVER_ID",
    "PREDICTOR_ARTIFACT_VERSION",
    "BatchPromptQualityPredictor",
    "BatchQualityPredictor",
    "BilinearPredictorArtifact",
    "BilinearQualityPredictor",
    "BilinearTrainingConfig",
    "CalibratedQualityPredictor",
    "GbmModel",
    "GbmNestedLodoWorkEstimate",
    "GbmQualityPredictor",
    "GbmTrainingConfig",
    "IsotonicCalibrator",
    "PerModelCalibratedQualityPredictor",
    "QualityPredictor",
    "RegressionStump",
    "RidgeSolver",
    "StaticQualityPredictor",
    "estimate_nested_lodo_gbm_work",
    "fit_calibrated_bilinear",
    "fit_calibrated_bilinear_for_fold",
    "fit_calibrated_gbm",
    "fit_calibrated_gbm_for_fold",
    "preflight_gbm_fit",
    "preflight_nested_lodo_gbm",
    "resolve_ridge_solver",
    "training_data_sha256",
]
