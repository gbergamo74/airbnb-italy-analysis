import os
import pytest
from predictor import load_model

# Test unitario: il modello si carica?
def test_model_loadable_or_skip():
    model = load_model()
    if model is None:
        pytest.skip("ridge_pipeline.pkl non presente nell'ambiente CI")
    assert hasattr(model, "predict")
