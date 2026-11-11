"""Compatibility alias for the selected model's prediction module."""
from importlib import import_module
import sys
from ai_server.model_config import MODEL_NAME

sys.modules[__name__] = import_module(f"ai_server.classifiers.{MODEL_NAME}.prediction")
