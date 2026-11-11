"""Model suites run in separate processes and dependency environments."""
import os

if os.environ.get("SKYDETECT_MODEL", "rf") == "minirocket":
    collect_ignore = ["test_feature_core.py", "test_rule_filter.py", "test_rf_explain.py"]
else:
    collect_ignore = ["minirocket"]
