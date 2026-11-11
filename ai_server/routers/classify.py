"""Keep model-specific HTTP contracts on the existing /classify paths."""
from importlib import import_module
from ai_server.model_config import MODEL_NAME

router = import_module(f"ai_server.classifiers.{MODEL_NAME}.router").router
