"""Legacy CLI/import compatibility; explicit model modules are also supported."""
from importlib import import_module
import runpy
import sys
from ai_server.model_config import MODEL_NAME

_target = f"ai_server.classifiers.{MODEL_NAME}.evaluate"
if __name__ == "__main__":
    runpy.run_module(_target, run_name="__main__")
else:
    sys.modules[__name__] = import_module(_target)
