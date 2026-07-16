<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native prepared-session protocol

Status: design-frozen experimental training protocol. It is separate from
`TRRIDG01`, is not used by routing or quickstart, and is not an approved release
accelerator until the repository records three-platform build, link, parity, and
full-shape resource evidence.

The protocol runs the complete prepared solve-and-score graph in one authenticated
child process. The input feature matrix remains file-backed: the child computes one
set of per-domain moments during the first feature scan, solves every canonical
training subset, then scans the features once more to fill every canonical score
block. It must not materialize the full feature matrix on the C heap.

## Identities

| Object | Identity |
|---|---|
| store container | `tierroute.prepared-store-file-f64le-v1` |
| session engine | `tierroute.prepared-session-c11-v1` |
| moment solver | `tierroute.prepared-moment-ridge-cholesky-c11-v1` |
| raw scorer | `tierroute.prepared-raw-dot-product-c11-v1` |
| result container | `tierroute.prepared-session-result-f64le-v1` |

The fixed magics are `TRPSTO01`, `TRPSES01`, and `TRPRES01`. These identities do
not reinterpret the in-memory prepared-store identities or the single-problem
`TRRIDG01` protocol. All integers are unsigned little-endian. All numeric cells are
finite IEEE-754 binary64 with positive zero as the only zero encoding. Text is
strict UTF-8. No C struct layout, platform-sized integer, padding byte with a
nonzero value, or trailing byte is accepted.

## Prepared store file version 1

The header is exactly 472 bytes. Seven-element arrays use zero for positions at or
above the declared domain count.

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | magic `TRPSTO01` |
| 8 | 4 | version `1` |
| 12 | 4 | flags `0` |
| 16 | 8 | header bytes, exactly `472` |
| 24 | 8 | exact file bytes |
| 32 | 8 | domain count `D`, 4 through 7 |
| 40 | 8 | row count `N` |
| 48 | 8 | universal feature count `d` |
| 56 | 8 | target count `M` |
| 64 | 8 | universal surface width, exactly `12` |
| 72 | 8 | row-key section offset, exactly `472` |
| 80 | 8 | row-key section bytes |
| 88 | 8 | domain-index section offset |
| 96 | 8 | domain-index bytes, exactly `N` |
| 104 | 8 | feature section offset, 8-byte aligned |
| 112 | 8 | feature bytes, exactly `8*N*d` |
| 120 | 8 | target section offset |
| 128 | 8 | target bytes, exactly `8*N*M` |
| 136 | 32 | prepared graph identity SHA-256 |
| 168 | 32 | caller-pinned source-fit SHA-256 |
| 200 | 32 | logical in-memory store SHA-256 |
| 232 | 32 | embedding snapshot SHA-256, or 32 zero bytes |
| 264 | 32 | embedding identity SHA-256, or 32 zero bytes |
| 296 | 32 | sorted model-catalogue SHA-256 |
| 328 | 32 | SHA-256 of every byte from offset 472 through EOF |
| 360 | 56 | seven domain row counts as `uint64` |
| 416 | 56 | seven domain active-tag masks as `uint64` |

The row-key section contains exactly `N` consecutive records:

```text
uint16 example_id_utf8_bytes
byte[example_id_utf8_bytes] example_id
byte[32] prompt_sha256
```

IDs are nonempty, at most 4,096 UTF-8 bytes, contain at least one byte other than
ASCII space, tab, LF, VT, FF, or CR, and are strictly increasing by Python Unicode
string order in the builder. Prompt digests are raw 32-byte values. The domain-index
section contains one `uint8` per row. Zero padding extends its end to
the next 8-byte boundary. Features and targets are row-major binary64 matrices. The
feature section immediately follows the padding; the target section immediately
follows the features; EOF immediately follows the targets.

The first three feature columns are nonnegative raw continuous surface values. The
next nine columns are binary (`has_code`, `has_math`, then the fixed seven-tag
catalogue). Remaining columns are caller-precomputed embeddings. The child verifies
the declared per-domain counts and active-tag masks while scanning.

The whole-file SHA-256 is an external credential and is never read from a sibling
`.sha256` file. The embedded payload digest is corruption evidence, not an
authentication credential. A trusted receipt binds the whole-file, source-fit,
logical-store, and optional embedding-snapshot digests.

The three store catalogue digests use the same unambiguous length framing as the
in-memory prepared identities. Every token is
`uint32 ASCII-label-bytes || label || uint64 payload-bytes || payload`; a text payload
is UTF-8 and an integer payload is one little-endian `uint64`. The first token is
always the `namespace` text token.

- `tierroute.prepared-store-file-graph-identity-v1` frames the graph algorithm ID,
  ordered domain names and counts, feature count, and target count.
- `tierroute.prepared-store-file-embedding-identity-v1` frames provider, model ID,
  revision, pooling, one-byte normalize flag, and asset-manifest SHA-256. It is zero
  only for a surface-only store.
- `tierroute.prepared-store-file-model-catalogue-v1` frames the count and every sorted
  model ID.

## Session request version 1

The request is a 160-byte header followed by one exact prepared store file.

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | magic `TRPSES01` |
| 8 | 4 | version `1` |
| 12 | 4 | flags `0` |
| 16 | 32 | unpredictable request nonce |
| 48 | 32 | exact store-file SHA-256 |
| 80 | 32 | exact authenticated child-binary SHA-256 |
| 112 | 8 | exact request bytes |
| 120 | 8 | exact expected success-result bytes |
| 128 | 8 | finite positive ridge binary64 |
| 136 | 24 | zero reserved bytes |
| 160 | variable | exact `TRPSTO01` file |

The caller descriptor-stably opens a regular non-symlink source, preflights its
fixed header, streams it once into an owner-only private request while hashing it,
checks exact length and stable metadata, and never reopens the source path. It also
snapshots and authenticates the child binary. The child receives the request file
as its already-open standard input in one invocation.

## Session result version 1

The result begins with a 448-byte header. A success result then contains all
coefficient records followed by all score records in graph order. A structured
failure ends after its header.

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | magic `TRPRES01` |
| 8 | 4 | version `1` |
| 12 | 4 | status |
| 16 | 32 | request nonce echo |
| 48 | 32 | store-file SHA-256 echo |
| 80 | 32 | child-binary SHA-256 echo |
| 112 | 8 each | `D`, `N`, `d`, `M` |
| 144 | 8 | coefficient-record count |
| 152 | 8 | score-record count |
| 160 | 8 | score row memberships |
| 168 | 8 | coefficient section bytes |
| 176 | 8 | score section bytes |
| 184 | 8 | exact result bytes |
| 192 | 8 | ridge echo |
| 200 | 8 | statistics work units |
| 208 | 8 | solve work units |
| 216 | 8 | score work units |
| 224 | 8 | modeled peak C-heap bytes |
| 232 | 32 | store payload SHA-256 echo |
| 264 | 32 | logical store SHA-256 echo |
| 296 | 32 | source-fit SHA-256 echo |
| 328 | 32 | embedding snapshot SHA-256 echo |
| 360 | 32 | model-catalogue SHA-256 echo |
| 392 | 32 | prepared graph identity SHA-256 echo |
| 424 | 8 | input authentication/validation bytes scanned |
| 432 | 8 | output numeric cells validated |
| 440 | 8 | file-backed input bytes |

Status values are `0` success, `1` protocol/header, `2` bounds/resource, `3`
numeric/input, `4` allocation, `5` solve/residual, and `6` I/O/internal. A success
requires status zero and process exit zero. A structured failure requires a known
nonzero status, nonzero process exit, and a header-only result. Every contradiction,
unknown status, absent header, or malformed echo fails closed.

Each coefficient record is:

```text
uint32 subset_index
uint32 reserved_zero
uint64 subset_domain_mask
uint64 training_row_count
uint64 active_tag_mask
uint64 active_feature_count
uint64 record_payload_bytes
f64[3] continuous_means
f64[3] continuous_scales
f64[M] intercepts
f64[M*active_feature_count] target-major weights
```

Each score record is:

```text
uint32 block_index
uint32 training_subset_index
uint32 scored_domain_index
uint32 reserved_zero
uint64 row_count
uint64 record_payload_bytes
f64[row_count*M] row-major scores
```

Training subsets follow the existing graph order: omit 3 domains in lexicographic
combination order, then omit 2, then omit 1. A score record follows for every omitted
domain of each subset, in ascending domain order. Continuous columns use population
scales `sqrt(centered_sum_squares / training_rows)`, with scale one for a zero
diagonal. Binary/tag and embedding columns are unscaled; only tags observed in a
training subset are active. One Cholesky factorization is shared by all targets in a
subset.

Every solved target is checked against its normal equation before intercept recovery:

```text
||A*w - b||inf <= 4096 * DBL_EPSILON * (active_width + 1)
                    * max(1, ||A||inf * ||w||inf + ||b||inf)
```

The norms and residual are accumulated in C `long double`, which a conforming target
may implement with the same precision as `double`. This is a corruption/solver-failure
gate, not the separate cross-implementation parity tolerance.

## Aggregate admission

Both Python and C use checked unsigned arithmetic before mmap, large allocation, or
child launch. Version-1 maxima are:

| Resource | Maximum |
|---|---:|
| store file | 512 MiB |
| result file | 128 MiB |
| modeled C heap | 512 MiB |
| private request plus result scratch | 1 GiB |
| total numeric work | 200,000,000,000 units |
| child timeout | 3,600 seconds |
| stderr | 16 KiB plus a local truncation marker |

For each of the canonical subsets, `w` is its active feature width. Admission sums
domain-statistics work, `w^3 + 2*M*w^2 + M*w` solve work, and every score row's
`M*w` dot-product work. It separately reports file-backed input bytes, input bytes
scanned for authentication/validation, modeled C heap, private disk scratch,
coefficient bytes, score bytes, and output-validation cells.
These are conservative deterministic units, not wall-clock or peak-RSS measurements.

The deterministic input-scan receipt is

```text
store_file_bytes + row_key_bytes + domain_index_bytes + alignment_padding
                 + target_bytes + 2*feature_bytes
```

because the payload hash is updated during the whole-store authentication scan. The
output-validation receipt is
`sum_subsets(6 + M + M*w) + score_row_memberships*M` binary64 cells.

## Trust and claim boundary

The adapter rejects path replacement, wrong externally supplied digests, malformed
or overlong streams, nonce/shape/config/binary lineage mismatch, contradictory
status/exit pairs, timeout, crash, and bounded-output overflow. POSIX owner-only temp
directories and ordinary Windows user temp ACLs do not sandbox a malicious process
running as the same user. Mapped drives and mounted network filesystems are caller
responsibilities.

Parity is tolerance-based. Byte-identical coefficients across compiler/CPU platforms
are not promised. Full RouterBench/bge-m3 execution, quality retention, cost savings,
or production readiness remain unproven until separately measured and recorded.
