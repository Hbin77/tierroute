# SPDX-License-Identifier: Apache-2.0
"""Finite resource contract for version-1 replay JSON datasets."""

# The planned RouterBench conversion is about 131 MiB compact or 161 MiB indented.
MAX_REPLAY_DATASET_BYTES = 256 * 1024 * 1024

# Version 1 is tied to the three public BudgetTier values. Subsets remain valid.
MAX_REPLAY_TIERS = 3
MAX_REPLAY_EXAMPLES = 100_000
MAX_REPLAY_OUTCOMES_PER_EXAMPLE = 4_096
MAX_REPLAY_TOTAL_OUTCOMES = 1_000_000

# LODO repeats every example across every domain's train/test partition. Bound that
# downstream amplification at the adapter boundary, before typed construction.
MAX_REPLAY_DOMAINS = 4_096
MAX_REPLAY_LODO_MEMBERSHIPS = 1_000_000
# Policy cross-fitting nests an inner LODO inside every outer-domain fold. The planned
# 34,778-example, seven-domain RouterBench shape needs 1,252,008 memberships.
MAX_REPLAY_NESTED_LODO_MEMBERSHIPS = 2_000_000
# The reference trainer currently extracts M targets with a linear outcome lookup for
# each of M models across N*D inner-LODO/final-fit example memberships.
MAX_REPLAY_TRAINING_OUTCOME_SCANS = 100_000_000

MAX_REPLAY_METADATA_TEXT_BYTES = 16 * 1024
MAX_REPLAY_PROMPT_TEXT_BYTES = 1024 * 1024
MAX_REPLAY_OUTPUT_TEXT_BYTES = 1024 * 1024
# Core exact costs support 100,000 decimal positions; this leaves grammar headroom.
MAX_REPLAY_COST_TEXT_BYTES = 128 * 1024

# CPython guarantees decimal integer conversion at this width even when an embedding
# application configures the interpreter's minimum non-zero int-string limit.
MAX_REPLAY_JSON_NUMBER_CHARACTERS = 640
# One schema version, three tier weights, and one quality for every possible outcome.
MAX_REPLAY_JSON_NUMBER_TOKENS = MAX_REPLAY_TOTAL_OUTCOMES + MAX_REPLAY_TIERS + 1
MAX_REPLAY_JSON_NESTING_DEPTH = 16
MAX_REPLAY_JSON_OBJECT_FIELDS = 7

# At simultaneous collection maxima, version 1 needs fewer than 9.71 million strings
# and 6.51 million opening-container/comma tokens, including optional quoted costs.
MAX_REPLAY_JSON_STRING_TOKENS = 10_000_000
MAX_REPLAY_JSON_STRUCTURE_TOKENS = 7_000_000
# A 1 MiB ASCII value can expand sixfold when every byte uses a JSON escape.
MAX_REPLAY_JSON_STRING_CHARACTERS = 6 * 1024 * 1024 + 2
