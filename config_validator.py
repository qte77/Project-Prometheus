"""
config_validator.py — schema-checked weights validation for Xenarch config files.

config_iter2.json / config_iter3.json legitimately target different pipeline
versions' scoring-weight schemas: mk17 (xenarch_pipeline.py, 'density' key) and
mk20 (xenarch_mk20_script.py, 'latent' key). This validates a config's
'combined_weights' dict against a *named* schema rather than assuming one
script version, so it doesn't misreport "drift" for a config that is correctly
aimed at the older pipeline.

Schema key sets below are copied from source and must be kept in sync by hand
if those scripts change (importlib-loading xenarch_mk20_script.py to derive
METRIC_KEYS live would pull in flask/flask_cors/loguru/torch, which this
validator has no other reason to require):
    mk20: xenarch_mk20_script.py:143 (METRIC_KEYS) / :148-154 (COMBINED_WEIGHTS)
    mk17: xenarch_pipeline.py:31-33 (DEFAULT_COMBINED_WEIGHTS)

Usage:
    python config_validator.py --schema mk20|mk17 <file.json> [<file2.json> ...]
"""
from __future__ import annotations

import argparse
import json
import sys

from pydantic import RootModel

SCHEMAS: dict[str, set] = {
    "mk20": {"mse", "latent", "contextual", "gradient", "edge"},
    "mk17": {"mse", "density", "contextual", "gradient", "edge"},
}

# config_iter2.json's combined_weights sums to 1.05, not exactly 1.0 — a
# real, currently-valid config, so the tolerance has to accommodate it.
WEIGHT_SUM_TOLERANCE = 0.10


class SchemaMismatchError(ValueError):
    """Weight dict's keys don't match the named schema's expected key set."""


class WeightSumError(ValueError):
    """Weight dict's values don't sum to ~1.0."""


class RawWeights(RootModel[dict[str, float]]):
    """Structural check only: a dict of string keys to float values."""


def validate_weights(weights: dict[str, float], schema: str) -> None:
    """Raise SchemaMismatchError/WeightSumError/ValueError, or return None if valid."""
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}, choose from {sorted(SCHEMAS)}")

    validated = RawWeights(weights).root  # pydantic: raises on non-numeric values

    expected = SCHEMAS[schema]
    actual = set(validated)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SchemaMismatchError(
            f"weights keys {sorted(actual)} don't match '{schema}' schema "
            f"{sorted(expected)} (missing={missing}, extra={extra})"
        )

    total = sum(validated.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise WeightSumError(
            f"weights sum to {total:.3f}, expected ~1.0 (+/- {WEIGHT_SUM_TOLERANCE})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schema", required=True, choices=sorted(SCHEMAS))
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    ok = True
    for path in args.files:
        with open(path) as f:
            data = json.load(f)
        weights = data.get("combined_weights")
        if weights is None:
            print(f"{path}: no 'combined_weights' key found", file=sys.stderr)
            ok = False
            continue
        try:
            validate_weights(weights, schema=args.schema)
            print(f"{path}: OK ({args.schema})")
        except (SchemaMismatchError, WeightSumError, ValueError) as e:
            print(f"{path}: FAIL ({args.schema}) — {e}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
