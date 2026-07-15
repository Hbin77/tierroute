# SPDX-License-Identifier: Apache-2.0
"""Package-level smoke tests."""

import tierroute


def test_package_has_version() -> None:
    assert tierroute.__version__ == "0.1.0"
