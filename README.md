# Xenarch — Planetary Technosignature Detection

Unsupervised anomaly detection for lunar and planetary surface imagery. A
variational autoencoder trains exclusively on natural terrain, then flags
whatever it reconstructs poorly — landing sites, rovers, other human-made
hardware — as a candidate technosignature.

## Why this exists

Human-made structures on another world's surface are rare, small, and by
definition don't look like the geology around them. There aren't enough
labeled anomalies to train a classifier on. So Xenarch trains only on what
natural terrain looks like, and treats high reconstruction error as the
signal: the model has never seen anything like this chip before.

## Goals

- Detect known human artifacts (Apollo landing sites) in orbital imagery
  without ever training on a labeled anomaly.
- Keep false positives from natural look-alikes — crater rims, boulder
  fields, shadow patterns — low enough for the output to be worth reviewing.
- Stay reproducible: same training corpus and config should give the same
  score every time.

## Current pipeline

`xenarch_mk20_script.py` is the active implementation: a VAE trained
against a fixed natural-terrain baseline (`training data/`), servable either
as a Flask web app or run headless against a folder of imagery.

    python xenarch_mk20_script.py --run [DIR]   # headless: train + score a folder
    python xenarch_mk20_script.py               # serve the web app (Flask)
    gunicorn xenarch_mk20_script:app            # production (Procfile-managed)

Full CLI flags and environment variables are documented in that script's own
module docstring.

`xenarch_mk3_script.py` through `xenarch_mk19_script.py` are version
history, not the current interface — each is a snapshot of one design
iteration, useful for comparing against an older result but not the script
to run by default.

## Setup

    uv sync                       # core deps (numpy/scipy/flask/pydantic/rasterio/...)
    uv sync --extra torch         # + PyTorch, for the gradient-based VAE path
                                   #   (optional -- mk20 falls back to a NumPy-only
                                   #   scorer without it)
    uv sync --extra server        # + gunicorn, for production serving
    uv sync --extra finetune      # + optuna, for finetune20.py's hyperparameter search
    python xenarch_mk20_script.py

## Tooling

- **`config_validator.py`** — validates `config_iter*.json` weight schemas before a
  run picks up the wrong one:

      python config_validator.py --schema mk20|mk17 config_iter2.json config_iter3.json

- **`eval_scoring.py`** — fast/small-scale proof that mk20's scoring logic separates
  known technosignatures from natural terrain (see its own module docstring for what
  this does and doesn't claim):

      python eval_scoring.py --positive "Test data" --negative "training data" \
          --out eval_result.json

- **`xenarch_mk3_script.py`**'s `TechnosignatureDB` reads its Postgres connection from
  `TECHNOSIG_DB_NAME`, `TECHNOSIG_DB_USER`, `TECHNOSIG_DB_PASSWORD`, `TECHNOSIG_DB_HOST`,
  `TECHNOSIG_DB_PORT` (falling back to the generic Postgres dev default when unset).

## Data

Training imagery comes from the Lunar Reconnaissance Orbiter (LROC) and Mars
Reconnaissance Orbiter (HiRISE) cameras. Two harvesters populate it:

- `lroc_fetch.py` — natural-terrain reference corpus (the negative set).
- `training data/download-curated-img.py` — curated HiRISE/CTX imagery.
- `apollo16_agent.py` — Apollo 16 descent-stage imagery, a known-truth positive
  control for `eval_scoring.py` (`Test data/`). Run `python apollo16_agent.py
  --self-test` for an offline sanity check of its geolocation/ranking logic.

Reference portals for finding more source imagery by hand:

- [ISSDC PRADAN — Chandrayaan-2 browse](https://pradan.issdc.gov.in/ch2/protected/browse.xhtml)
- [ISSDC PRADAN — TMC2](https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=tmc2)
- [PDS Imaging Atlas — HiRISE EDR search](https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.instrument=HIRISE&gather.common.product_type=EDR)
- [PDS Imaging Atlas — MGS/MOC search](https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.spacecraft=mars_global_surveyor&gather.common.instrument=MOC) (lower resolution)
- [PDS Imaging Atlas — LRO search](https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.mission=lro) (rough fit, not a primary source)

## Learn more

- **Architecture, math, and design rationale:** see `Architecture
  Documentation`.
- **Research writeup:** `Xenarch_Mk13_White_Paper*.docx` — note the
  validation claims there are for Mk13, not yet re-verified against the
  current Mk20 head.

## License

This README used to assert MIT with no `LICENSE` file backing it up.
`[TODO: owner — confirm the license and add the file]`

## Contact

`[TODO: owner]`
