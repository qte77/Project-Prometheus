"""RED-first test for eval_scoring.py's combined mk20 anomaly score.

Fast/small-scale proof of the scoring logic (see eval_scoring.py's module
docstring), not a production run: PyTorch is not installed in this sandbox,
so `xenarch_mk20_script` falls back to its own NumpyAnomalyScorer (its own
words: "when torch is unavailable this is the whole scorer"). There is no
gradient-based VAE training here.

Asserts that "Apollo 11 landing site.png" (Test data/ -- the Mk13 white
paper's own canonical positive control) scores ABOVE THE MEDIAN of the
known-natural baseline (training data/) once both are run through mk20's own
robust median/MAD normalization + combined-score weighting. This is
deliberately NOT a claim that every Apollo chip beats every natural chip --
it does not (a handful of naturally rough/high-contrast natural terrain
chips score higher than some Apollo chips; see eval_scoring.py's CLI output
for the full breakdown). The claim under test is narrower and true: the
known technosignature sits above the calibrated center of the natural
distribution, not merely "some" natural chips.
"""
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import eval_scoring  # noqa: E402

POSITIVE_DIR = REPO_ROOT / "Test data"
NEGATIVE_DIR = REPO_ROOT / "training data"
POSITIVE_FILE = "Apollo 11 landing site.png"


def test_known_apollo_chip_scores_above_natural_median():
    scorer, stats = eval_scoring.build_baseline(NEGATIVE_DIR)

    positive_scores = eval_scoring.score_dir(POSITIVE_DIR, scorer, stats)
    negative_scores = eval_scoring.score_dir(NEGATIVE_DIR, scorer, stats)

    assert positive_scores, "expected at least one positive (Apollo) chip"
    assert negative_scores, "expected at least one negative (natural) chip"

    apollo = next(s for s in positive_scores if s["file"] == POSITIVE_FILE)
    natural_median = statistics.median(s["combined"] for s in negative_scores)

    assert apollo["combined"] > natural_median
