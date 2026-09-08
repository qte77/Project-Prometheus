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

There's no `requirements.txt` in the repo yet, so the install step below
doesn't work today — noting it here rather than pretending otherwise:

    [TODO: pip install -r requirements.txt, or `uv sync` once pyproject.toml lands]
    python xenarch_mk20_script.py

## Data

Training imagery comes from the Lunar Reconnaissance Orbiter (LROC) and Mars
Reconnaissance Orbiter (HiRISE) cameras. Two harvesters populate it:

- `lroc_fetch.py` — natural-terrain reference corpus (the negative set).
- `training data/download-curated-img.py` — curated HiRISE/CTX imagery.

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
