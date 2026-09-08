"""
RED-first tests for config_validator.py.

config_iter2.json/config_iter3.json legitimately target the mk17 pipeline's
weight schema (`density` key), not mk20's (`latent` key) — these tests assert
that distinction is what the validator actually enforces, not "config_iter2
is broken."
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config_validator import SchemaMismatchError, WeightSumError, validate_weights

CONFIG_ITER2 = HERE / "config_iter2.json"


def _combined_weights(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["combined_weights"]


def test_config_iter2_valid_for_mk17_schema():
    # config_iter2.json correctly targets xenarch_pipeline.py's (mk17) schema —
    # this must NOT raise.
    validate_weights(_combined_weights(CONFIG_ITER2), schema="mk17")


def test_config_iter2_invalid_for_mk20_schema():
    # Same file, checked against mk20's schema (which uses 'latent' instead of
    # 'density') — this is the real, demonstrable mismatch.
    with pytest.raises(SchemaMismatchError):
        validate_weights(_combined_weights(CONFIG_ITER2), schema="mk20")


def test_weights_not_summing_to_one_fails_regardless_of_schema():
    bad_mk17 = {"mse": 0.1, "density": 0.1, "contextual": 0.1, "gradient": 0.1, "edge": 0.1}
    with pytest.raises(WeightSumError):
        validate_weights(bad_mk17, schema="mk17")

    bad_mk20 = {"mse": 0.1, "latent": 0.1, "contextual": 0.1, "gradient": 0.1, "edge": 0.1}
    with pytest.raises(WeightSumError):
        validate_weights(bad_mk20, schema="mk20")


def test_unknown_schema_name_rejected():
    with pytest.raises(ValueError):
        validate_weights(_combined_weights(CONFIG_ITER2), schema="mk99")
