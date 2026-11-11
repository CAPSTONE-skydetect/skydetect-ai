"""Compatibility exports. Shared A/B contracts are independent of the C model."""
from ai_server.contracts.common import *
from ai_server.model_config import MODEL_NAME

if MODEL_NAME == "rf":
    from ai_server.classifiers.rf.contracts import *
else:
    from ai_server.classifiers.minirocket.contracts import *
