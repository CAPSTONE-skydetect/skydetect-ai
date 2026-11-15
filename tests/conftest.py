"""Model suites run in separate processes and dependency environments."""
import os

_RF_ONLY = ["test_feature_core.py", "test_rule_filter.py", "test_rf_explain.py"]
_MODEL = os.environ.get("SKYDETECT_MODEL", "rf")
if _MODEL == "minirocket":
    collect_ignore = _RF_ONLY + ["multirocket"]
elif _MODEL == "multirocket":
    # multirocket 이미지에는 MiniRocket 단일 모델 파일이 없다.
    collect_ignore = _RF_ONLY + ["minirocket"]
else:
    collect_ignore = ["minirocket", "multirocket"]
