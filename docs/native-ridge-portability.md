# SPDX-License-Identifier: Apache-2.0

# Native source-portability gate

The `native-source-portability` CI job is configured to compile both project-owned C11
sources on `macos-latest` and `windows-latest`, run `tests/test_native_ridge.py` plus
the prepared file/session suites, and audit one ephemeral local executable per source
on each host. No executable is uploaded or included in the wheel. The workflow
definition is not passing-run evidence; the new prepared-session candidate's first
macOS/Windows CI result remains pending until a run finishes successfully.

When green, this job proves only that the current source candidates compile and pass
their protocol and numerical tests in those runner environments. It is not a
released-artifact audit:
the runner images and compiler installations are not pinned binary inputs, the
ephemeral hashes are not release hashes, and no platform executable receives license or
SBOM approval from this job.

The build helper emits source and executable SHA-256 values. CI parses that JSON and the
audit CLI requires both values; a missing, malformed, uppercase, or mismatched digest is
fatal. The audit copies each original through `lstat` → `O_NOFOLLOW` open → `fstat` into
an owner-only temporary directory, checks bounded size/hash and final inode metadata,
then runs every platform tool only against those private snapshots. The dependency
report, import report, binary hash, and source lexical scan therefore refer to the same
manifest-bound bytes even if an original path is replaced after the snapshot.
Candidate binaries share the adapter/build limit of 16 MiB, and UNC or device-style
binary and source paths are refused before inspection.

## Expected platform imports

- macOS must link only `/usr/lib/libSystem.B.dylib`. Its complete symbol-only
  `nm -u -j` report is printed. Every nonempty row must be one canonical symbol; format
  drift, partial rows, and direct process-creation, network, `dlopen`, or `dlsym`
  imports are rejected.
- Windows is built with MSVC `/MT` after `vswhere.exe` locates the installed toolchain
  and `VsDevCmd.bat` establishes the x64 environment. `dumpbin /DEPENDENTS` must list
  only `KERNEL32.dll`; this excludes dynamic MSVC/UCRT runtime DLLs. The complete
  `dumpbin /IMPORTS` report is printed. The parser requires the full executable
  preamble, dependency/import markers, per-DLL metadata, named symbol rows, and Summary
  section; dependency and import DLL sets must agree. Unknown, partial, duplicate, or
  ordinal-only rows and direct process-creation or network imports are rejected. Kernel
  APIs used by the statically linked runtime for memory, standard I/O, errors, and normal
  termination are expected; their exact set can vary with the hosted MSVC toolset and is
  preserved in each CI log.

The static Windows CRT may itself import `GetProcAddress` or `LoadLibraryExW` for
operating-system compatibility thunks. The audit records those dynamic-resolution
imports but does not reject them or attribute them to project code from the import table
alone. To keep that limitation explicit, the same gate rejects direct
process/network/dynamic-resolution identifiers in the reviewed project C source. This
is a lexical backstop, not a proof against obfuscated calls, manual PE traversal, or a
compromised toolchain; the structured result therefore always records
`indirect_capability_absence_proven: false`.

The audit is deliberately fail-closed if a tool report is empty or unparseable. The
deny-list supplements, rather than replaces, source review: an import table cannot prove
the absence of every indirect operating-system capability.

## Out of scope

This gate does not build or approve a Linux-musl executable. It does not establish a
three-platform release matrix, reproducible compiler provenance, distributable binary
hashes, license closure, or an executable SBOM record. Those remain separate release
requirements. The dependency-free wheel must exclude both native sources; the source
distribution must contain exactly `native/tierroute_ridge.c` and
`native/tierroute_prepared.c` so downstream review and local compilation remain
possible. CI treats the working `native/` tree as an exact two-file allowlist, rejects
symbolic/hard links, executable mode and native/binary
suffixes in release archives, requires no exact `native` path segment anywhere in the
wheel, and verifies that the working source, sdist member, and SBOM SHA-256 are
identical.
