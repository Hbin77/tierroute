# SPDX-License-Identifier: Apache-2.0
"""Prove that installed runtime commands do not attempt network connections."""

from __future__ import annotations

import socket
import urllib.request

import pytest

from tierroute.cli import main


def test_route_evaluate_and_demo_never_open_a_socket(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    capsys.readouterr()
