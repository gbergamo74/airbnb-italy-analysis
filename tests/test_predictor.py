import pytest, os
from predictor import load_model, predict_one

def test_model_present_or_skip():
    model = load_model()
    if model is None:
        pytest.skip("No model in CI environment")
    out = predict_one({
        "room_type": "Private room",
        "number_of_reviews": 5,
        "availability_365": 120
    })
    assert isinstance(out, float)
