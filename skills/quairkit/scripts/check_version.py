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

r"""Fail closed unless the installed QuAIRKit distribution is exactly 0.5.1."""

from __future__ import annotations

import sys
from importlib import util
from importlib import metadata
from pathlib import Path


DISTRIBUTION_NAME = "quairkit"
EXPECTED_VERSION = "0.5.1"


class VersionGateError(RuntimeError):
    r"""Raised when the required QuAIRKit distribution is unavailable."""


def require_exact_version() -> tuple[str, Path]:
    try:
        distribution = metadata.distribution(DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError as exc:
        raise VersionGateError(
            f"QuAIRKit {EXPECTED_VERSION} is required, but the distribution is not installed."
        ) from exc

    detected = distribution.version
    distribution_root = Path(distribution.locate_file("")).resolve()
    if detected != EXPECTED_VERSION:
        raise VersionGateError(
            f"QuAIRKit {EXPECTED_VERSION} is required, but {detected} was detected at {distribution_root}."
        )
    spec = util.find_spec(DISTRIBUTION_NAME)
    if spec is None or spec.origin is None:
        raise VersionGateError("The installed QuAIRKit package cannot be resolved for import.")
    import_origin = Path(spec.origin).resolve()
    try:
        import_origin.relative_to(distribution_root)
    except ValueError as exc:
        raise VersionGateError(
            f"QuAIRKit import resolves to {import_origin}, outside the official installation at {distribution_root}."
        ) from exc
    return detected, import_origin


def main() -> int:
    try:
        detected, import_origin = require_exact_version()
    except VersionGateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"OK: QuAIRKit {detected} imports from {import_origin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
