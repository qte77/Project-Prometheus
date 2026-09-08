"""Wires apollo16_agent.py's own offline --self-test into the test suite.

The dependency audit for this PR flagged that nothing in CI exercises
apollo16_agent.py or lroc_fetch.py at all. This doesn't close that gap fully
(lroc_fetch.py still has zero coverage, and this is not network testing --
--self-test is explicitly offline/no-network per that script's own --help),
but it is a real, non-trivial check: apollo16_agent.py's internal PDS-label
parsing, tangent-plane geolocation math, and ranking logic all get exercised,
not just "does the file import."
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_apollo16_agent_self_test_passes():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "apollo16_agent.py"), "--self-test"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL PASS" in result.stdout
