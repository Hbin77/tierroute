# SPDX-License-Identifier: Apache-2.0
"""Resource contracts for portable predictor artifacts."""

MAX_PREDICTOR_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_PREDICTOR_MODELS = 4_096
MAX_PREDICTOR_TRAINING_DOMAINS = 4_096
MAX_PREDICTOR_NUMERIC_SCALARS = 1_000_000
# CPython guarantees decimal integer conversion at this width even when an embedding
# application configures the interpreter's minimum non-zero int-string limit.
MAX_PREDICTOR_JSON_NUMBER_CHARACTERS = 640
# Version 1 has five fixed numeric identity/count fields outside the model parameters.
MAX_PREDICTOR_JSON_NUMBER_TOKENS = MAX_PREDICTOR_NUMERIC_SCALARS + 5
MAX_PREDICTOR_METADATA_TEXT_BYTES = 4 * 1024
MAX_PREDICTOR_METADATA_TOTAL_BYTES = 1024 * 1024
MAX_PREDICTOR_CALIBRATOR_POINTS = MAX_PREDICTOR_NUMERIC_SCALARS // 2
# The simultaneous maxima (4,096 models/domains/tags) need 28,703 JSON strings.
MAX_PREDICTOR_JSON_STRING_TOKENS = 32_768
# A 4 KiB metadata value can expand sixfold when every byte uses a JSON escape.
MAX_PREDICTOR_JSON_STRING_CHARACTERS = 6 * MAX_PREDICTOR_METADATA_TEXT_BYTES + 2
MAX_PREDICTOR_JSON_NESTING_DEPTH = 32
# One million numeric parameters plus maximum maps, arrays, and metadata stay below this.
MAX_PREDICTOR_JSON_STRUCTURE_TOKENS = 1_100_000
