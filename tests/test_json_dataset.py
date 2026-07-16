# SPDX-License-Identifier: Apache-2.0
"""Tests for the bounded version-1 replay JSON trust boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import tierroute.adapters.json_dataset as json_dataset
import tierroute.adapters.resource_limits as replay_limits
import tierroute.cli as cli
from tierroute.adapters import bundled_synthetic_path, load_evaluation_dataset
from tierroute.core import BudgetTier


def _bundled_payload() -> dict[str, Any]:
    return json.loads(bundled_synthetic_path().read_text(encoding="utf-8"))


def _write_payload(
    tmp_path: Path,
    payload: object,
    *,
    ensure_ascii: bool = False,
) -> Path:
    destination = tmp_path / "replay.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=ensure_ascii, separators=(",", ":")),
        encoding="utf-8",
    )
    return destination


def _schema_object(payload: dict[str, Any], context: str) -> dict[str, Any]:
    if context == "root":
        return payload
    if context == "tier":
        return payload["tier_specs"][0]
    if context == "example":
        return payload["examples"][0]
    if context == "outcome":
        return payload["examples"][0]["outcomes"][0]
    raise AssertionError(f"unknown test context {context!r}")


def test_bundled_dataset_is_complete_and_explicitly_synthetic() -> None:
    dataset = load_evaluation_dataset()

    assert bundled_synthetic_path().is_file()
    assert dataset.license == "Apache-2.0"
    assert "not benchmark evidence" in dataset.provenance
    assert dataset.domain_labels_are_observable is True
    assert len(dataset.examples) == 8
    assert {example.domain for example in dataset.examples} == {
        "general",
        "code",
        "math",
        "science",
    }
    assert [spec.tier for spec in dataset.tier_specs] == list(BudgetTier)
    assert dataset.examples[0].router_metadata["domain"] == dataset.examples[0].domain
    assert "example_id" not in dataset.examples[0].router_metadata


def test_bundled_dataset_bytes_are_stable() -> None:
    payload = bundled_synthetic_path().read_bytes()

    assert len(payload) == 7_395
    assert hashlib.sha256(payload).hexdigest() == (
        "e4c4a04ff6151828a426f387f7225c7fd65a25ee5ca257506182076be65cdea9"
    )


def test_bundled_costs_are_exact_and_model_catalogue_is_stable() -> None:
    dataset = load_evaluation_dataset()

    for example in dataset.examples:
        assert [outcome.model_id for outcome in example.outcomes] == [
            "swift",
            "steady",
            "expert",
        ]
        assert [outcome.cost for outcome in example.outcomes] == [
            Decimal("0.20"),
            Decimal("0.60"),
            Decimal("1.00"),
        ]


def test_reader_accepts_exact_byte_limit_and_rejects_limit_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "replay.json"
    payload = bundled_synthetic_path().read_bytes()
    source.write_bytes(payload)
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_DATASET_BYTES", len(payload))

    assert load_evaluation_dataset(source).name == "tierroute synthetic smoke dataset"

    monkeypatch.setattr(json_dataset, "MAX_REPLAY_DATASET_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="exceeds 7,394 bytes"):
        load_evaluation_dataset(source)


def test_reader_counts_multibyte_utf8_bytes_not_python_characters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _bundled_payload()
    payload["examples"][0]["prompt"] = "한글" * 300
    source = _write_payload(tmp_path, payload)
    byte_count = len(source.read_bytes())
    assert byte_count > len(source.read_text(encoding="utf-8"))
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_DATASET_BYTES", byte_count)

    assert load_evaluation_dataset(source).examples[0].prompt == "한글" * 300

    monkeypatch.setattr(json_dataset, "MAX_REPLAY_DATASET_BYTES", byte_count - 1)
    with pytest.raises(ValueError, match=f"exceeds {byte_count - 1:,} bytes"):
        load_evaluation_dataset(source)


def test_oversized_file_is_rejected_before_json_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "replay.json"
    payload = bundled_synthetic_path().read_bytes()
    source.write_bytes(payload)
    parsed = False

    def forbidden_json_loads(*args: object, **kwargs: object) -> object:
        nonlocal parsed
        del args, kwargs
        parsed = True
        raise AssertionError("oversized input reached json.loads")

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise AssertionError("oversized input reached os.read")

    monkeypatch.setattr(json_dataset, "MAX_REPLAY_DATASET_BYTES", len(payload) - 1)
    monkeypatch.setattr(json_dataset.json, "loads", forbidden_json_loads)
    monkeypatch.setattr(json_dataset.os, "read", forbidden_read)

    with pytest.raises(ValueError, match="exceeds"):
        load_evaluation_dataset(source)
    assert parsed is False


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"\xff", "not valid UTF-8"),
        (b"\xef\xbb\xbf{}", "not valid strict JSON"),
    ],
)
def test_reader_normalizes_invalid_utf8_and_bom(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    source = tmp_path / "replay.json"
    source.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        load_evaluation_dataset(source)


def test_reader_rejects_non_regular_files_without_blocking(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        load_evaluation_dataset(tmp_path)

    if not hasattr(os, "mkfifo"):
        return
    fifo = tmp_path / "replay.fifo"
    os.mkfifo(fifo)
    script = (
        "import sys\n"
        "from tierroute.adapters import load_evaluation_dataset\n"
        "try:\n"
        "    load_evaluation_dataset(sys.argv[1])\n"
        "except ValueError as error:\n"
        "    print(error)\n"
        "else:\n"
        "    raise SystemExit('FIFO was accepted')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(fifo)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0
    assert "regular file" in completed.stdout


def test_reader_accepts_symlink_to_stable_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(bundled_synthetic_path().read_bytes())
    link = tmp_path / "replay.json"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    assert load_evaluation_dataset(link).name == "tierroute synthetic smoke dataset"


def test_reader_fails_closed_when_path_is_replaced_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("Windows may reject replacement of an open file before adapter validation")
    source = tmp_path / "replay.json"
    replacement = tmp_path / "replacement.json"
    source.write_bytes(bundled_synthetic_path().read_bytes())
    changed = _bundled_payload()
    changed["name"] = "replacement"
    replacement.write_text(json.dumps(changed), encoding="utf-8")
    real_read = os.read
    replaced = False

    def replace_then_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replacement.replace(source)
            replaced = True
        return real_read(descriptor, size)

    monkeypatch.setattr(json_dataset.os, "read", replace_then_read)

    with pytest.raises(ValueError, match="changed while reading"):
        load_evaluation_dataset(source)


def test_reader_detects_same_inode_growth_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "replay.json"
    source.write_bytes(bundled_synthetic_path().read_bytes())
    real_read = os.read
    changed = False

    def grow_after_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, size)
        if not changed:
            with source.open("ab") as stream:
                stream.write(b" ")
            changed = True
        return chunk

    monkeypatch.setattr(json_dataset.os, "read", grow_after_read)

    with pytest.raises(ValueError, match="changed while reading"):
        load_evaluation_dataset(source)


def test_reader_detects_same_size_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "replay.json"
    source.write_bytes(bundled_synthetic_path().read_bytes())
    real_read = os.read
    changed = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, size)
        if not changed:
            with source.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"[")
                stream.flush()
                os.fsync(stream.fileno())
            changed = True
        return chunk

    monkeypatch.setattr(json_dataset.os, "read", mutate_after_read)

    with pytest.raises(ValueError, match="changed while reading"):
        load_evaluation_dataset(source)


@pytest.mark.parametrize(
    "constant_name, document, exact_count",
    [
        ("MAX_REPLAY_JSON_NESTING_DEPTH", "[[[[[0]]]]]", 5),
        ("MAX_REPLAY_JSON_STRING_TOKENS", '["a","b"]', 2),
        ("MAX_REPLAY_JSON_STRUCTURE_TOKENS", "[{},{}]", 4),
    ],
)
def test_json_structural_preflight_exact_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    document: str,
    exact_count: int,
) -> None:
    monkeypatch.setattr(json_dataset, constant_name, exact_count)
    json_dataset._parse_strict_json(document)

    monkeypatch.setattr(json_dataset, constant_name, exact_count - 1)
    with pytest.raises(ValueError, match="JSON"):
        json_dataset._parse_strict_json(document)


def test_json_object_and_string_lexical_limits_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_document = json.dumps({f"k{index}": index for index in range(7)})
    real_json_loads = json_dataset.json.loads
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_JSON_OBJECT_FIELDS", 7)
    assert len(json_dataset._parse_strict_json(object_document)) == 7  # type: ignore[arg-type]

    parsed = False

    def forbidden_json_loads(*args: object, **kwargs: object) -> object:
        nonlocal parsed
        del args, kwargs
        parsed = True
        raise AssertionError("oversized object reached json.loads")

    monkeypatch.setattr(json_dataset, "MAX_REPLAY_JSON_OBJECT_FIELDS", 6)
    monkeypatch.setattr(json_dataset.json, "loads", forbidden_json_loads)
    with pytest.raises(ValueError, match="field limit"):
        json_dataset._parse_strict_json(object_document)
    assert parsed is False

    string_document = json.dumps("x" * 20)
    monkeypatch.setattr(json_dataset.json, "loads", real_json_loads)
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_JSON_OBJECT_FIELDS", 7)
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_JSON_STRING_CHARACTERS", len(string_document))
    assert json_dataset._parse_strict_json(string_document) == "x" * 20
    monkeypatch.setattr(
        json_dataset,
        "MAX_REPLAY_JSON_STRING_CHARACTERS",
        len(string_document) - 1,
    )
    with pytest.raises(ValueError, match="string exceeds"):
        json_dataset._parse_strict_json(string_document)


def test_json_number_width_and_count_limits_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_integer = "1" * replay_limits.MAX_REPLAY_JSON_NUMBER_CHARACTERS
    assert json_dataset._parse_strict_json(exact_integer) == int(exact_integer)
    with pytest.raises(ValueError, match="lexical limit"):
        json_dataset._parse_strict_json(exact_integer + "1")

    document = "[1,2,3]"
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_JSON_NUMBER_TOKENS", 3)
    assert json_dataset._parse_strict_json(document) == [1, 2, 3]
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_JSON_NUMBER_TOKENS", 2)
    with pytest.raises(ValueError, match="number-token limit"):
        json_dataset._parse_strict_json(document)


@pytest.mark.parametrize(
    "document",
    [
        "9" * (replay_limits.MAX_REPLAY_JSON_NUMBER_CHARACTERS + 1),
        "0." + "1" * replay_limits.MAX_REPLAY_JSON_NUMBER_CHARACTERS,
        "1e" + "9" * replay_limits.MAX_REPLAY_JSON_NUMBER_CHARACTERS,
    ],
)
def test_oversized_json_number_is_rejected_before_json_parser(
    monkeypatch: pytest.MonkeyPatch,
    document: str,
) -> None:
    parsed = False

    def forbidden_json_loads(*args: object, **kwargs: object) -> object:
        nonlocal parsed
        del args, kwargs
        parsed = True
        raise AssertionError("oversized number reached json.loads")

    monkeypatch.setattr(json_dataset.json, "loads", forbidden_json_loads)

    with pytest.raises(ValueError, match="lexical limit"):
        json_dataset._parse_strict_json(document)
    assert parsed is False


@pytest.mark.parametrize("document", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_json_nonstandard_and_nonfinite_numbers_are_rejected(document: str) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        json_dataset._parse_strict_json(document)


def test_duplicate_keys_are_rejected_at_root_and_nested_outcome(tmp_path: Path) -> None:
    original = bundled_synthetic_path().read_text(encoding="utf-8")
    duplicate_root = original.replace(
        '  "name": "tierroute synthetic smoke dataset",',
        '  "schema_version": "shadow",',
        1,
    )
    root_source = tmp_path / "duplicate-root.json"
    root_source.write_text(duplicate_root, encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON"):
        load_evaluation_dataset(root_source)

    duplicate_outcome = original.replace(
        '"model_id": "swift"',
        '"model_id": "shadow", "model_id": "swift"',
        1,
    )
    outcome_source = tmp_path / "duplicate-outcome.json"
    outcome_source.write_text(duplicate_outcome, encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        load_evaluation_dataset(outcome_source)


@pytest.mark.parametrize(
    "context, missing_field",
    [
        ("root", "domain_labels_are_observable"),
        ("tier", "weight"),
        ("example", "prompt"),
        ("outcome", "quality"),
    ],
)
def test_schema_rejects_missing_fields(
    tmp_path: Path,
    context: str,
    missing_field: str,
) -> None:
    payload = _bundled_payload()
    del _schema_object(payload, context)[missing_field]

    with pytest.raises(ValueError, match="fields mismatch"):
        load_evaluation_dataset(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("context", ["root", "tier", "example", "outcome"])
def test_schema_rejects_unknown_fields(tmp_path: Path, context: str) -> None:
    payload = _bundled_payload()
    target = _schema_object(payload, context)
    if context == "root":
        target["unexpected"] = target.pop("name")
    else:
        target["unexpected"] = "ignored-before-v0.1"

    with pytest.raises(ValueError, match="fields mismatch"):
        load_evaluation_dataset(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", 2])
def test_schema_version_requires_exact_integer_one(
    tmp_path: Path,
    schema_version: object,
) -> None:
    payload = _bundled_payload()
    payload["schema_version"] = schema_version

    with pytest.raises(ValueError, match="integer 1"):
        load_evaluation_dataset(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("visibility", [0, 1, "true", None])
def test_domain_visibility_requires_an_exact_boolean(
    tmp_path: Path,
    visibility: object,
) -> None:
    payload = _bundled_payload()
    payload["domain_labels_are_observable"] = visibility

    with pytest.raises(ValueError, match="must be a boolean"):
        load_evaluation_dataset(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("collection", ["tier_specs", "examples", "outcomes"])
def test_schema_collections_require_json_arrays(tmp_path: Path, collection: str) -> None:
    payload = _bundled_payload()
    if collection == "outcomes":
        payload["examples"][0]["outcomes"] = {}
    else:
        payload[collection] = {}

    with pytest.raises(ValueError, match="must be a list"):
        load_evaluation_dataset(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    "field_path, value",
    [
        (("tier_specs", 0, "weight"), True),
        (("tier_specs", 0, "weight"), "0.5"),
        (("tier_specs", 0, "weight"), None),
        (("examples", 0, "outcomes", 0, "quality"), True),
        (("examples", 0, "outcomes", 0, "quality"), "0.85"),
        (("examples", 0, "outcomes", 0, "quality"), None),
    ],
)
def test_quality_and_weight_reject_implicit_number_coercions(
    tmp_path: Path,
    field_path: tuple[object, ...],
    value: object,
) -> None:
    root = _bundled_payload()
    payload: Any = root
    for key in field_path[:-1]:
        payload = payload[key]
    payload[field_path[-1]] = value

    with pytest.raises(ValueError, match="JSON number"):
        load_evaluation_dataset(_write_payload(tmp_path, root))


@pytest.mark.parametrize("cost", [1, " 1", "1_0", "NaN", "Infinity", "-1"])
def test_costs_require_bounded_decimal_strings(tmp_path: Path, cost: object) -> None:
    payload = _bundled_payload()
    payload["examples"][0]["outcomes"][0]["cost"] = cost

    with pytest.raises(ValueError, match=r"string|decimal syntax|non-negative"):
        load_evaluation_dataset(_write_payload(tmp_path, payload))


def test_quoted_cost_requires_an_exact_decimal_string(tmp_path: Path) -> None:
    payload = _bundled_payload()
    payload["examples"][0]["outcomes"][0]["quoted_cost"] = 0.2

    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_evaluation_dataset(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("cost", ["-0", "-0.0", "+0", "00", "01", ".5", "1.", "1e+2"])
def test_cost_grammar_preserves_finite_decimal_compatibility(
    tmp_path: Path,
    cost: str,
) -> None:
    payload = _bundled_payload()
    payload["examples"][0]["outcomes"][0]["cost"] = cost

    loaded = load_evaluation_dataset(_write_payload(tmp_path, payload))

    assert loaded.examples[0].outcomes[0].cost == Decimal(cost)


def test_optional_quote_fallback_and_explicit_domain_visibility_are_stable(
    tmp_path: Path,
) -> None:
    payload = _bundled_payload()
    assert "quoted_cost" not in payload["examples"][0]["outcomes"][0]
    payload["domain_labels_are_observable"] = False
    dataset = load_evaluation_dataset(_write_payload(tmp_path, payload))

    assert dataset.examples[0].candidate_models[0].cost == dataset.examples[0].outcomes[0].cost
    assert dataset.examples[0].router_metadata == {}


@pytest.mark.parametrize(
    "constant_name, exact_count, error_fragment",
    [
        ("MAX_REPLAY_TIERS", 3, "tier_specs"),
        ("MAX_REPLAY_EXAMPLES", 8, "examples"),
        ("MAX_REPLAY_OUTCOMES_PER_EXAMPLE", 3, "outcomes"),
        ("MAX_REPLAY_TOTAL_OUTCOMES", 24, "aggregate"),
        ("MAX_REPLAY_DOMAINS", 4, "domain"),
        ("MAX_REPLAY_LODO_MEMBERSHIPS", 32, "LODO"),
        ("MAX_REPLAY_TRAINING_OUTCOME_SCANS", 288, "outcome-scan"),
        ("MAX_REPLAY_NESTED_LODO_MEMBERSHIPS", 72, "nested-LODO"),
    ],
)
def test_collection_and_lodo_limits_have_exact_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    exact_count: int,
    error_fragment: str,
) -> None:
    monkeypatch.setattr(json_dataset, constant_name, exact_count)
    assert len(load_evaluation_dataset().examples) == 8

    monkeypatch.setattr(json_dataset, constant_name, exact_count - 1)
    with pytest.raises(ValueError, match=error_fragment):
        load_evaluation_dataset()


@pytest.mark.parametrize(
    "field_path, constant_name",
    [
        (("name",), "MAX_REPLAY_METADATA_TEXT_BYTES"),
        (("examples", 0, "prompt"), "MAX_REPLAY_PROMPT_TEXT_BYTES"),
        (("examples", 0, "outcomes", 0, "output"), "MAX_REPLAY_OUTPUT_TEXT_BYTES"),
    ],
)
def test_semantic_text_limits_count_utf8_bytes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[object, ...],
    constant_name: str,
) -> None:
    payload: Any = _bundled_payload()
    value = "한" * 400
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    source = _write_payload(tmp_path, payload)
    byte_count = len(value.encode("utf-8"))
    monkeypatch.setattr(json_dataset, constant_name, byte_count)

    load_evaluation_dataset(source)

    monkeypatch.setattr(json_dataset, constant_name, byte_count - 1)
    with pytest.raises(ValueError, match="text limit"):
        load_evaluation_dataset(source)


def test_cost_text_limit_is_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _bundled_payload()
    cost = "0." + "0" * 1_000 + "1"
    payload["examples"][0]["outcomes"][0]["cost"] = cost
    source = _write_payload(tmp_path, payload)
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_COST_TEXT_BYTES", len(cost))

    assert load_evaluation_dataset(source).examples[0].outcomes[0].cost == Decimal(cost)

    monkeypatch.setattr(json_dataset, "MAX_REPLAY_COST_TEXT_BYTES", len(cost) - 1)
    with pytest.raises(ValueError, match="text limit"):
        load_evaluation_dataset(source)


def test_escaped_lone_surrogate_is_rejected_semantically(tmp_path: Path) -> None:
    payload = _bundled_payload()
    payload["examples"][0]["outcomes"][0]["output"] = "\ud800"
    source = _write_payload(tmp_path, payload, ensure_ascii=True)

    with pytest.raises(ValueError, match="valid Unicode"):
        load_evaluation_dataset(source)


def test_limits_cover_measured_planned_routerbench_shape() -> None:
    assert replay_limits.MAX_REPLAY_DATASET_BYTES >= 161 * 1024 * 1024
    assert replay_limits.MAX_REPLAY_TIERS >= 3
    assert replay_limits.MAX_REPLAY_EXAMPLES >= 34_778
    assert replay_limits.MAX_REPLAY_OUTCOMES_PER_EXAMPLE >= 11
    assert replay_limits.MAX_REPLAY_TOTAL_OUTCOMES >= 34_778 * 11
    assert replay_limits.MAX_REPLAY_LODO_MEMBERSHIPS >= 34_778 * 7
    assert replay_limits.MAX_REPLAY_TRAINING_OUTCOME_SCANS >= 34_778 * 7 * 11**2
    assert replay_limits.MAX_REPLAY_NESTED_LODO_MEMBERSHIPS >= 34_778 * 6**2
    assert replay_limits.MAX_REPLAY_PROMPT_TEXT_BYTES >= 5_052
    assert replay_limits.MAX_REPLAY_OUTPUT_TEXT_BYTES >= 16_101


@pytest.mark.parametrize("command", ["route", "evaluate", "train"])
def test_all_data_cli_paths_fail_at_the_same_bounded_loader_before_downstream_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    source = tmp_path / "replay.json"
    payload = bundled_synthetic_path().read_bytes()
    source.write_bytes(payload)
    monkeypatch.setattr(json_dataset, "MAX_REPLAY_DATASET_BYTES", len(payload) - 1)

    def forbidden_downstream(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("unsafe replay input reached downstream work")

    monkeypatch.setattr(cli, "route_prompt", forbidden_downstream)
    monkeypatch.setattr(cli, "evaluate_six_baselines", forbidden_downstream)
    monkeypatch.setattr(cli, "fit_calibrated_bilinear", forbidden_downstream)
    arguments = {
        "route": ["route", "hello", "--data", str(source)],
        "evaluate": ["evaluate", "--data", str(source)],
        "train": [
            "train",
            "--data",
            str(source),
            "--output",
            str(tmp_path / "predictor.json"),
        ],
    }[command]

    with pytest.raises(ValueError, match=f"exceeds {len(payload) - 1:,} bytes"):
        cli.main(arguments)
    assert not (tmp_path / "predictor.json").exists()
