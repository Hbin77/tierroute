<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dependency license audit

This record applies tierroute's project rule literally: a selected distribution must
not contain a GPL, LGPL, or AGPL family component, even when an exception or dynamic
linking would make ordinary Apache-2.0 distribution legally possible. This is a
competition compliance policy, not a claim that every rejected package is legally
incompatible with Apache-2.0.

Top-level package metadata is not sufficient evidence. Each review below inspected the
actual wheel contents, nested metadata, bundled license documents, direct requirements,
and native-library relationships. Runtime packages remain unapproved until every
supported platform has equivalent evidence.

## Approved build backend

`flit_core==3.12.0` is approved for build and editable-install time only.

| Artifact | SHA-256 |
|---|---|
| `flit_core-3.12.0-py3-none-any.whl` | `e7a0304069ea895172e3c7bb703292e992c5d1555dd1233ab7b5621b5b69e62c` |
| `flit_core-3.12.0.tar.gz` | `18f63100d6f94385c6ed57a72073443e1a71a4acb4339491615d0f16d6ff01b2` |

The backend is BSD-3-Clause, declares no distribution dependency, and contains no
native library. It vendors Tomli 1.2.3 under MIT, a packaging-derived PEP 440 regex
under its inline BSD notice, and an identifier table generated from the CC0-1.0 SPDX
License List data. Wheel, sdist, src-layout discovery, package data, console entry
point, wheel install, editable install, and offline smoke were verified. The runtime
wheel does not depend on or contain flit_core.

## Reviewed permissive license evidence

Some PSF-family license texts contain GPL compatibility discussion and historical
Python distribution references, so keyword scanning alone produces false positives.
The deep gate exempts only these exact reviewed document hashes:

| Distribution and evidence | Evidence SHA-256 |
|---|---|
| `typing_extensions==4.16.0` `licenses/LICENSE` | `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` |
| `pip==26.1.2` vendored `distlib==0.4.0` `LICENSE.txt` | `808e10c8a6ab8deb149ff9b3fb19f447a808094606d712a9ca57fead3552599d` |

The audited `pip-26.1.2-py3-none-any.whl` has SHA-256
`382ff9f685ee3bc25864f820aa50505825f10f5458ffff07e30a6d96e5715cab`.
Any evidence byte change falls back to the normal fail-closed scan. Python 3.10 CI
verifies the installed typing-extensions evidence, while every CI job runs the deep
gate over the pinned pip evidence.

## Rejected candidates

### setuptools 83.0.0

The official wheel
`setuptools-83.0.0-py3-none-any.whl` has SHA-256
`29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3`.
Although its top-level metadata reports MIT, it includes executable
`setuptools/_vendor/autocommand/*.py`; the adjacent nested metadata declares LGPLv3
and the bundled `LICENSE` contains LGPL-3.0. tierroute therefore replaced setuptools
instead of relying on the top-level `pip-licenses` result.

### NumPy 2.2.6

NumPy 2.2.6 was audited because it supports the project's Python 3.10 and 3.12 matrix.

| Wheel | SHA-256 |
|---|---|
| CPython 3.10, manylinux2014 x86-64 | `fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915` |
| CPython 3.12, manylinux2014 x86-64 | `fd83c01228a688733f1ded5201c678f0c53ecc1006ffbc404db9f7a899ac6249` |
| CPython 3.10, macOS 14 ARM64 | `37e990a01ae6ec7fe7fa1c26c55ecb672dd98b19c3d0e1d1f326fa13cb38d163` |
| CPython 3.12, macOS 14 ARM64 | `894b3a42502226a1cac872f840030665f33326fc3dac8e57c607905773cdcde3` |

Both Linux wheels physically bundle `libgfortran` under
GPL-3.0-with-GCC-exception and `libquadmath` under LGPL-2.1-or-later. OpenBLAS links to
libgfortran, which links to libquadmath. The macOS wheels use Apple Accelerate rather
than bundled dylibs, but a macOS-only selection would not satisfy the Linux CI and
distribution contract. NumPy is rejected unless the owner explicitly changes the
strict policy and a new platform-complete audit is approved.

### PyTorch 2.13.0 CPU

| Wheel | SHA-256 |
|---|---|
| CPython 3.12, manylinux 2.28 x86-64 | `4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2` |
| CPython 3.12, macOS 14 ARM64 | `2fe228aba290d14b9f31b049be550dbd469c3fd3013d7a19705b30454da97027` |
| CPython 3.12, Windows x86-64 | `a8b450c1e58e5800e5b4691dac412f8d2d65a1dc3298166f91596603a3531e6f` |

Every wheel contains a GPLv3 license under Kineto's vendored CPR test tree. In
addition, the required setuptools distribution has the vendored LGPL component above.
The Linux CPU dependency graph occupies roughly 202 MB compressed and 741 MB installed;
the ordinary PyPI Linux wheel also selects CUDA packages, so it is not an acceptable
minimal offline training backend.

### tinygrad 0.11.0

The pure-Python MIT wheel has SHA-256
`b901d98880f04ad9f796734a013151ba851dbb9a340f1a516099adcd6fd3b3e3` and the sdist has
SHA-256 `d9d468a55906cc49a1b4df5b69be78a58ab4d15714b7f238f2b0876d2bc09bc1`.
However, its Linux CPU path unconditionally loads `libgcc_s.so.1`, which is governed by
GPLv3 plus the GCC Runtime Library Exception. It also has no direct Cholesky/linear-solve
API, has backend-specific float64 limitations, and does not provide the offline/network
contract required here. It is rejected under the same literal policy.

## Reproduction pattern

Download without dependencies, hash before extraction, then inspect nested metadata,
license files, native binaries, and link requirements. For example:

```bash
python -m pip download --only-binary=:all: --no-deps \
  --dest /tmp/tierroute-audit flit_core==3.12.0
shasum -a 256 /tmp/tierroute-audit/*
unzip -q /tmp/tierroute-audit/*.whl -d /tmp/tierroute-audit/unpacked
rg -n -i 'AGPL|LGPL|GPL|GNU .*General Public License' \
  /tmp/tierroute-audit/unpacked
```

For native wheels, also enumerate shared libraries and inspect their dynamic link
tables (`objdump -p` on ELF, `otool -L` on Mach-O, and an equivalent PE import-table
tool on Windows). Record the exact artifact hashes and platform tags in this document
before approval.
