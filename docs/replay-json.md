<!-- SPDX-License-Identifier: Apache-2.0 -->

# Version-1 replay JSON boundary

## Purpose

Replay JSON is an adapter-owned interchange format for the bundled demo and local
offline experiments. It is not the unresolved SK Telecom schema. `route`, `evaluate`,
`train`, and `demo` all call the same `load_evaluation_dataset()` boundary before
command-specific work. The loader never performs network access and accepts only a
stable local regular file.

The bundled `synthetic.json` remains exactly 7,395 bytes with SHA-256
`e4c4a04ff6151828a426f387f7225c7fd65a25ee5ca257506182076be65cdea9`. It is illustrative
project-authored data, not benchmark evidence.

## Descriptor and parser contract

The reader opens the caller-selected path once with nonblocking and close-on-exec flags,
then treats that descriptor as authoritative. A symlink is allowed only when its opened
target is a regular file. Size is checked with `fstat` before allocation, at most the
limit plus one byte is read in bounded chunks, and device/inode/size/modification/change
metadata plus the observed byte count must remain stable. A concurrent replacement or
mutation that changes the opened descriptor's evidence fails closed. Bytes are decoded
as strict UTF-8; a BOM is not silently removed.

Before `json.loads`, a lexical pass bounds nesting, object members, strings, numbers,
and opening-container/comma tokens without constructing decoded JSON values. Parsing
then rejects duplicate keys, `NaN`, `Infinity`, overflowing binary64 numbers, wide or
excessive numeric tokens, recursion failures, and malformed JSON. Every decoded text
field is encoded back to strict UTF-8 so escaped lone surrogates also fail.

## Exact version-1 schema

Every object rejects unknown fields. Root, tier, and example fields are all required.
Only `quoted_cost` is optional; when absent it is exactly the same string as `cost`.

| Object | Required fields | Optional fields |
|---|---|---|
| root | `schema_version`, `name`, `license`, `provenance`, `domain_labels_are_observable`, `tier_specs`, `examples` | none |
| tier | `tier`, `budget_limit`, `weight` | none |
| example | `example_id`, `prompt`, `domain`, `outcomes` | none |
| outcome | `model_id`, `output`, `cost`, `quality` | `quoted_cost` |

`schema_version` must be the exact JSON integer `1`, not `true`, `1.0`, or a string.
Domain visibility must be a JSON boolean. Tier weight and outcome quality must be JSON
integers or floats that fit finite IEEE-754 binary64; booleans and numeric strings are
not coerced.

Budgets and costs intentionally remain exact decimal strings. Their accepted lexical
grammar is `[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?`; the core then
requires the represented value to be finite, non-negative, and inside its exact-cost
range. This preserves finite forms such as `-0.0`, `.5`, `1.`, and `1e+2` without
allowing whitespace, underscores, `NaN`, or `Infinity`.

## Resource limits

The constants below live in `adapters/resource_limits.py`. They are a simultaneous,
fail-closed contract, not targets or evidence that a dataset is licensed.

| Resource | Version-1 limit |
|---|---:|
| UTF-8 file bytes | 268,435,456 (256 MiB) |
| tiers | 3 |
| examples | 100,000 |
| outcomes per example | 4,096 |
| outcomes across the dataset | 1,000,000 |
| unique domains | 4,096 |
| outer-LODO memberships, `examples × domains` | 1,000,000 |
| nested-LODO memberships, `examples × (domains − 1)²` | 2,000,000 |
| ordinary metadata or ID text | 16,384 UTF-8 bytes |
| one prompt | 1,048,576 UTF-8 bytes |
| one output | 1,048,576 UTF-8 bytes |
| one cost string | 131,072 UTF-8 bytes |
| JSON nesting depth | 16 |
| fields in one JSON object | 7 |
| characters in one source JSON string token | 6,291,458 |
| JSON string tokens | 10,000,000 |
| characters in one JSON number token | 640 |
| JSON number tokens | 1,000,004 |
| opening-container plus comma tokens | 7,000,000 |

The file and token limits are also the aggregate lexical bounds; there is no
environment variable or CLI option that disables them. LODO limits cover downstream
fold-reference amplification before typed replay objects or predictor fits are built.

The planned RouterBench mapping used to choose headroom has 34,778 examples, 11 models,
7 domains, and 382,558 outcomes. A measured conversion was about 130.69 MiB compact or
160.77 MiB indented, with maximum prompt/output sizes of 5,052/16,101 UTF-8 bytes. These
measurements justify limits only; RouterBench is not bundled and its redistribution
license remains unresolved.

## Migration and official-data rule

Do not add an unlimited flag, permissive fallback, field alias, or automatic schema
detection. If authenticated official data exceeds a limit, first record its checksum
and measured byte/token/count/max-text shape. Then either raise the adapter-local
constant with exact-limit and limit-plus-one tests when the document is genuinely the
same schema, or add a separate official adapter/new `schema_version` when fields or
semantics differ. Budget scope, call history, and cascade semantics remain outside this
format until the organizer confirms them.
