"""Exercise the adapter against the separate deployed MiniRocket runtime."""

from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("mode", ["unsupported_rate", "no_windows"])
def test_candidate_abstention_contract_in_minirocket_runtime(mode):
    repo = Path(__file__).resolve().parents[2]
    runtime = repo.parent / "skydetect-ai-minirocket-runtime"
    python = runtime / ".venv/bin/python"
    if not python.exists():
        pytest.skip("separate MiniRocket runtime is not installed")
    code = '''
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from minirocket_stack import StackMiniRocketClassifier
import research.trajectory_sequence as sequence
class Track:
    def model_dump(self, **kwargs): return {}
def reject(*args, **kwargs):
    raise ValueError("Source FPS below supported 15 Hz")
sequence.window_track = reject if sys.argv[3] == "unsupported_rate" else lambda *a, **k: ([], [{"reason": "long_gap"}])
classifier = StackMiniRocketClassifier.__new__(StackMiniRocketClassifier)
result = classifier.predict(Track())
assert result.label == "uncertain" and result.decision_score is None
if sys.argv[3] == "unsupported_rate":
    assert result.abstain_reason == "invalid_input"
    assert "15 Hz" in result.abstain_detail
else:
    assert result.abstain_reason == "insufficient_observation"
    assert result.window_rejections == {"long_gap": 1}
'''
    subprocess.run([str(python), "-c", code, str(runtime), str(repo / "research"), mode],
                   cwd=repo, check=True, capture_output=True, text=True, timeout=60)
