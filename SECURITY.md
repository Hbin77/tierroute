<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security policy

## Supported versions

tierroute is pre-1.0. Until the first stable release, security fixes target the current
default branch. Older commits and development branches are not separately supported.

## Report a vulnerability privately

Do not open a public issue or discussion for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/Hbin77/tierroute/security/advisories/new)
so the report and any proposed fix remain private until coordinated disclosure is safe.

Include only what is needed to reproduce and assess the issue:

- the affected tierroute version or commit;
- impact and the conditions required to trigger it;
- a minimal reproduction using synthetic data;
- relevant platform and Python-version details;
- suggested mitigations, if known.

Never send credentials, private prompts, restricted challenge data, unlicensed model
weights, or arbitrary pickle files. Replace sensitive inputs with the smallest
synthetic example that preserves the behavior.

The maintainer will acknowledge the report as soon as practicable, coordinate testing
and disclosure through the private advisory, and credit reporters who wish to be named.

## Optional native training boundary

The experimental C11 ridge sidecar is not a sandbox for arbitrary executables. A caller
must explicitly supply an absolute path and exact lowercase SHA-256 for the intended
project-owned binary. The digest authenticates only the caller-selected byte sequence;
it is not an approval, source-provenance attestation, import audit, or proof of network
absence. Native CLI output therefore does not assert end-to-end no-network execution:
it reports `network_used=null`, `python_orchestration_network_used=false`, and
`native_binary_audit=caller-responsibility-unapproved`.

tierroute rejects empty or larger-than-16-MiB binaries at the path, open-descriptor, and
streamed-byte boundaries. Training preflight authenticates the executable before an
embedding provider is called, and every solve repeats authentication while making its
private snapshot so replacement after preflight still fails. The adapter uses no shell
or PATH lookup, passes a restricted environment, caps request/work/allocation/response/
stderr resources, and validates the versioned response and process exit together. It
never downloads or discovers a binary automatically. Its configured timeout covers the
child process after launch; bounded binary authentication and request serialization
happen before launch and are not wall-clock-limited by that timeout.

Paths beginning with `//` or `\\` are rejected on every host to exclude UNC and device-
style spellings. A mapped drive or mounted network filesystem cannot be recognized
portably; callers must ensure all compiler, executable, data, and output paths are truly
local when making an offline claim.

Only binaries built from the reviewed project source are in scope. On POSIX, timeout
cleanup kills the sidecar process group. On Windows, cleanup kills the main process;
source review confirms that the project C does not create children. The portability gate
rejects direct named process/network imports and direct identifiers in the project C,
but permitted runtime dynamic-resolution imports mean it does not prove the absence of
every indirect capability. Do not use the adapter as a general third-party plugin runner.
Generalizing it would require a Windows Job Object or equivalent descendant-containment
design and a new threat review.
