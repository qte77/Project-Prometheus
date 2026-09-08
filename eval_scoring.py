"""
eval_scoring.py -- small/fast proof-of-concept anomaly-scoring harness.

This is a FAST, SMALL-SCALE proof of the combined-anomaly-score logic
already built into `xenarch_mk20_script.py` -- it is NOT a production run
and makes no claim about production-grade detection accuracy.

Reuses `xenarch_mk20_script.py`'s existing scoring code via `importlib`,
the same pattern `finetune20.py` already uses to import that script as a
library without triggering its Flask `app.run()` (guarded by
`if __name__ == "__main__"` in that file, so importing it only builds the
`app` object -- nothing is served here).

Why NumpyAnomalyScorer, not the torch VAE: PyTorch is not installed in this
sandbox, so `xenarch_mk20_script`'s own `HAS_TORCH` guard is False and mk20
already falls back to its own `NumpyAnomalyScorer` ("when torch is
unavailable this is the whole scorer" -- that class's own docstring). There
is no gradient-based VAE training here; "training" in this harness means
fitting that scorer's robust median/MAD reference statistics on a folder of
known-natural images, which is mk20's own calibration step for that scorer.
"Epochs" do not apply in this code path.

Chip extraction: each source image is reduced to ONE native-resolution
CENTER CROP (mk20's "crop" mode), not mk20's real sliding-window scene
tiling ("extract" mode, used by its Flask `run_analysis` pipeline). A
sliding-window prototype was tried and discarded for this harness only
because of wall-clock budget (~3.5 min for the full training+Test data sets
vs ~20s for center-crop) -- see docs/plans/0001-xenarch-modernize.md's
feat/eval-harness row for the timing budget this was built against.

CLI:
    python eval_scoring.py --positive "Test data" --negative "training data" \
        --out eval_result.json
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

# ── load the Mk20 pipeline as a library (see finetune20.py for the same
#    pattern): importing it only builds the Flask app object (app.run() is
#    guarded by `if __name__ == "__main__"` in that file), so nothing is
#    served here. ────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "xenarch_mk20_script", str(HERE / "xenarch_mk20_script.py"))
mk20 = importlib.util.module_from_spec(_spec)
sys.modules["xenarch_mk20_script"] = mk20
_spec.loader.exec_module(mk20)

CHIP_SIZE = mk20.CHIP_SIZE
TRAIN_EXTS = mk20.TRAIN_EXTS


def _center_crop(path: Path, chip_size: int = CHIP_SIZE) -> np.ndarray:
    """One (chip_size, chip_size) float32 chip per source file: a native-
    resolution center crop when the image is at least chip_size in both
    dimensions, else an area-resize fallback for undersized images. Uses
    only mk20's PUBLIC `load_image_as_array` so this harness doesn't couple
    to mk20's private/internal helpers (Giuly overwrites that whole file via
    web upload -- see the plan doc's watch-outs)."""
    arr = mk20.load_image_as_array(str(path))
    h, w = arr.shape
    if h >= chip_size and w >= chip_size:
        y0, x0 = (h - chip_size) // 2, (w - chip_size) // 2
        return np.ascontiguousarray(
            arr[y0:y0 + chip_size, x0:x0 + chip_size], dtype=np.float32)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((chip_size, chip_size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _iter_images(dir_path) -> list[Path]:
    root = Path(dir_path)
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in TRAIN_EXTS)


def load_chips(dir_path, chip_size: int = CHIP_SIZE) -> tuple[list[np.ndarray], list[str]]:
    """Load every image directly under dir_path as one center-crop chip
    each. Returns (chips, file names) in matching order."""
    files = _iter_images(dir_path)
    chips = [_center_crop(f, chip_size) for f in files]
    names = [f.name for f in files]
    return chips, names


def build_baseline(training_dir, chip_size: int = CHIP_SIZE):
    """Fit mk20's NumpyAnomalyScorer reference stats + combined-score
    normalization on `training_dir` (the known-natural baseline). This is
    the calibration/"training" step for mk20's numpy-fallback scorer -- see
    module docstring for why there is no gradient-based training here."""
    chips, _names = load_chips(training_dir, chip_size)
    if not chips:
        raise SystemExit(f"no training images found under {training_dir}")
    scorer = mk20.NumpyAnomalyScorer(reference_chips=chips)
    raw_scores = [scorer.score(c) for c in chips]
    stats = mk20.NumpyAnomalyScorer.fit_norm_stats(raw_scores)
    return scorer, stats


def score_dir(dir_path, scorer, stats, chip_size: int = CHIP_SIZE) -> list[dict]:
    """Score every image directly under dir_path against an already-fit
    baseline. Returns one dict per file with mk20's raw + normalized metrics
    plus the weighted 'combined' anomaly score (higher = farther from the
    natural baseline / more anomalous)."""
    chips, names = load_chips(dir_path, chip_size)
    raw = [scorer.score(c) for c in chips]
    normed = mk20.NumpyAnomalyScorer.apply_norm(raw, stats)
    for name, row in zip(names, normed, strict=True):
        row["file"] = name
        row.pop("feature_bbox", None)
    return normed


def _combined_stats(scores: list[dict]) -> dict[str, float]:
    vals = [s["combined"] for s in scores]
    return {
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fast/small-scale proof of mk20's combined anomaly-score "
                     "logic (NOT a production run): fits a baseline on "
                     "--negative (known-natural chips), then scores both "
                     "--positive (known-anomalous chips) and --negative "
                     "against it.")
    ap.add_argument("--positive", required=True,
                     help="dir of known-anomalous images (e.g. 'Test data')")
    ap.add_argument("--negative", required=True,
                     help="dir of known-natural images -- used both to fit "
                          "the baseline and as the negative comparison set "
                          "(e.g. 'training data')")
    ap.add_argument("--out", default="eval_result.json")
    ap.add_argument("--chip-size", type=int, default=CHIP_SIZE)
    args = ap.parse_args(argv)

    scorer, stats = build_baseline(args.negative, args.chip_size)
    positive_scores = score_dir(args.positive, scorer, stats, args.chip_size)
    negative_scores = score_dir(args.negative, scorer, stats, args.chip_size)

    result = {
        "positive_dir": str(args.positive),
        "negative_dir": str(args.negative),
        "positive": positive_scores,
        "negative": negative_scores,
        "positive_combined_stats": _combined_stats(positive_scores),
        "negative_combined_stats": _combined_stats(negative_scores),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))

    p, n = result["positive_combined_stats"], result["negative_combined_stats"]
    print(f"positive (n={len(positive_scores)}): mean={p['mean']:.4f} "
          f"median={p['median']:.4f} min={p['min']:.4f} max={p['max']:.4f}")
    print(f"negative (n={len(negative_scores)}): mean={n['mean']:.4f} "
          f"median={n['median']:.4f} min={n['min']:.4f} max={n['max']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
