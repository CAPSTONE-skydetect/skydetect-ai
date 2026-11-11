"""Legacy classifier imports, loaded lazily so each image needs one backend."""
from importlib import import_module

_EXPORTS = {
    "RFClassifier": "rf",
    "MiniRocketClassifier": "minirocket",
    "SequencePrediction": "minirocket",
}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"ai_server.classifiers.{_EXPORTS[name]}.classifier"), name)
