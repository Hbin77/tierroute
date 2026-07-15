# SPDX-License-Identifier: Apache-2.0
"""Prove that installed runtime commands do not attempt network connections."""

from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

import pytest

from tierroute.cli import main


def test_runtime_commands_never_open_a_socket(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def deny_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("runtime network access is prohibited")

    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(urllib.request, "urlopen", deny_network)

    assert main(["route", "offline prompt", "--tier", "fast", "--json"]) == 0
    assert main(["evaluate", "--json"]) == 0
    assert main(["demo"]) == 0
    artifact = tmp_path / "predictor.json"
    assert main(["train", "--output", str(artifact), "--json"]) == 0
    assert (
        main(
            [
                "route",
                "offline artifact prompt",
                "--artifact",
                str(artifact),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
