#!/usr/bin/env python3
# Copyright (c) 2026 QuAIR team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Recommend canonical QuAIRKit imports from the static API registry."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from check_version import EXPECTED_VERSION, VersionGateError, require_exact_version


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "references" / "imports.json"
REQUIRED_FIELDS = {
    "name",
    "qualified_name",
    "kind",
    "tier",
    "actual_import",
    "recommended_import",
    "preferred_usage",
    "alternatives",
    "notes",
}
ALLOWED_KINDS = {"module", "class", "function", "method", "constant"}
TIER_ORDER = {"root": 0, "circuit_method": 1, "submodule": 2, "low_level": 3}


class RegistryError(ValueError):
    r"""Raised when the checked-in import registry is malformed."""


def _validate_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RegistryError(f"symbols[{index}] must be an object")
    if set(record) != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - set(record))
        extra = sorted(set(record) - REQUIRED_FIELDS)
        raise RegistryError(f"symbols[{index}] fields mismatch: missing={missing}, extra={extra}")
    for field in ("name", "qualified_name", "kind", "tier", "recommended_import", "preferred_usage", "notes"):
        if not isinstance(record[field], str):
            raise RegistryError(f"symbols[{index}].{field} must be a string")
    if record["actual_import"] is not None and not isinstance(record["actual_import"], str):
        raise RegistryError(f"symbols[{index}].actual_import must be a string or null")
    if not isinstance(record["alternatives"], list) or not all(
        isinstance(item, str) for item in record["alternatives"]
    ):
        raise RegistryError(f"symbols[{index}].alternatives must be a list of strings")
    if record["kind"] not in ALLOWED_KINDS:
        raise RegistryError(f"symbols[{index}].kind is invalid: {record['kind']}")
    if record["tier"] not in TIER_ORDER:
        raise RegistryError(f"symbols[{index}].tier is invalid: {record['tier']}")
    if not record["name"] or not record["qualified_name"] or not record["recommended_import"]:
        raise RegistryError(f"symbols[{index}] contains an empty required value")
    return record


def load_registry(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RegistryError("registry schema_version must be 1")
    if payload.get("quairkit_version") != EXPECTED_VERSION:
        raise RegistryError(f"registry quairkit_version must be {EXPECTED_VERSION}")
    raw_symbols = payload.get("symbols")
    if not isinstance(raw_symbols, list):
        raise RegistryError("registry symbols must be a list")
    symbols = [_validate_record(record, index) for index, record in enumerate(raw_symbols)]
    qualified = [record["qualified_name"] for record in symbols]
    if qualified != sorted(qualified):
        raise RegistryError("registry symbols must be sorted by qualified_name")
    if len(qualified) != len(set(qualified)):
        raise RegistryError("registry contains duplicate qualified_name values")
    return symbols


def _rank(record: dict[str, Any]) -> tuple[int, str]:
    return TIER_ORDER[record["tier"]], record["qualified_name"].casefold()


def _deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({record["qualified_name"]: record for record in records}.values())


def resolve(query: str, symbols: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[str]]:
    exact_qualified = [record for record in symbols if record["qualified_name"] == query]
    if exact_qualified:
        return "matched", exact_qualified, []

    exact_name = sorted([record for record in symbols if record["name"] == query], key=_rank)
    if exact_name:
        status = "matched" if len(exact_name) == 1 else "ambiguous"
        return status, exact_name, []

    folded = query.casefold()
    casefold_matches = _deduplicate(
        [
            record
            for record in symbols
            if record["name"].casefold() == folded or record["qualified_name"].casefold() == folded
        ]
    )
    if casefold_matches:
        casefold_matches.sort(key=_rank)
        status = "matched" if len(casefold_matches) == 1 else "ambiguous"
        return status, casefold_matches, []

    label_to_qualified: dict[str, set[str]] = {}
    for record in symbols:
        for label in (record["name"], record["qualified_name"]):
            label_to_qualified.setdefault(label, set()).add(record["qualified_name"])
    close_labels = difflib.get_close_matches(query, sorted(label_to_qualified), n=8, cutoff=0.55)
    suggestions: list[str] = []
    for label in close_labels:
        for qualified_name in sorted(label_to_qualified[label]):
            if qualified_name not in suggestions:
                suggestions.append(qualified_name)
            if len(suggestions) == 5:
                return "unknown", [], suggestions
    return "unknown", [], suggestions


def result_payload(query: str, status: str, matches: list[dict[str, Any]], suggestions: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "query": query,
        "preferred": matches[0] if matches else None,
        "matches": matches,
        "suggestions": suggestions,
    }


def _print_record(record: dict[str, Any], prefix: str = "") -> None:
    print(f"{prefix}{record['qualified_name']} [{record['kind']}, {record['tier']}]")
    if record["actual_import"]:
        print(f"  Actual import: {record['actual_import']}")
    print(f"  Recommended import: {record['recommended_import']}")
    print(f"  Preferred usage: {record['preferred_usage']}")
    if record["notes"]:
        print(f"  Note: {record['notes']}")


def print_text(payload: dict[str, Any]) -> None:
    status = payload["status"]
    if status == "matched":
        _print_record(payload["preferred"])
        return
    if status == "ambiguous":
        print(f"Ambiguous query: {payload['query']}")
        print(f"Preferred candidate: {payload['preferred']['qualified_name']}")
        print("Candidates:")
        for record in payload["matches"]:
            _print_record(record, prefix="- ")
        return
    print(f"No import recommendation found for: {payload['query']}")
    if payload["suggestions"]:
        print("Did you mean:")
        for suggestion in payload["suggestions"]:
            print(f"- {suggestion}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Function, class, method, or qualified API name")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit stable JSON output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_exact_version()
    except VersionGateError as exc:
        print(f"Version gate failed: {exc}", file=sys.stderr)
        return 4
    try:
        symbols = load_registry()
    except RegistryError as exc:
        print(f"Registry error: {exc}", file=sys.stderr)
        return 3
    status, matches, suggestions = resolve(args.query, symbols)
    payload = result_payload(args.query, status, matches, suggestions)
    if args.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_text(payload)
    return {"matched": 0, "unknown": 1, "ambiguous": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
