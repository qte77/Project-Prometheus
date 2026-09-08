"""Imports xenarch_mk3_script.py once, with its heavy third-party deps
(geopandas, rasterio, psycopg2, torch, torchvision, sklearn, matplotlib,
seaborn, loguru, pandas, yaml) stubbed out, so test_mk3_credential.py can
exercise it without installing that legacy script's full dependency set.

mk3 is historical/frozen (mk20 is the active pipeline) and pulls in a heavy
geospatial/ML stack for what its own docstring calls a demo script; none of
that stack is otherwise part of this repo's dependencies (see
pyproject.toml, which scopes to the active mk20/finetune20/xenarch_pipeline/
lroc_fetch imports only).

IMPORTANT: the stubs are removed from sys.modules again immediately after
importing xenarch_mk3_script (its own already-bound references to them, e.g.
`xenarch_mk3_script.psycopg2`, are unaffected — Python keeps a module's own
attribute bindings regardless of later sys.modules changes). This must stay
scoped to "long enough to import mk3" and never become session-wide (e.g. a
pytest_configure hook that stubs for the whole test run) — other test files
in this directory (e.g. an eval-harness test that actually trains a small
model) need the REAL torch/numpy, and a global stub would silently hand them
a MagicMock instead of a working library.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_STUB_MODULES = [
    "pandas",
    "geopandas",
    "shapely",
    "shapely.geometry",
    "rasterio",
    "rasterio.windows",
    "rasterio.plot",
    "psycopg2",
    "psycopg2.extras",
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "torchvision",
    "sklearn",
    "sklearn.model_selection",
    "sklearn.metrics",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "tqdm",
    "loguru",
    "yaml",
]

_added = [name for name in _STUB_MODULES if name not in sys.modules]
for _name in _added:
    sys.modules[_name] = MagicMock(name=_name)

import xenarch_mk3_script  # noqa: E402  (imported once, while stubbed, on purpose)

for _name in _added:
    del sys.modules[_name]
