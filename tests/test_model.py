import joblib
import numpy as np
import pytest


# Load model and feature info once (important fix)
model = joblib.load("models/best_model.pkl")
features = joblib.load("models/features.pkl")
n_features = len(features)


def test_model_load():  # ensures your model file exists and loads correctly
    model = joblib.load("models/best_model.pkl")
    assert model is not None


def test_prediction_shape():  # verifies batch predictions return the expected shape
    sample = np.random.rand(1, n_features)
    pred = model.predict(sample)

    assert pred.shape == (1,)


def test_multiple_predictions():  # verifies batch prediction
    sample = np.random.rand(5, n_features)
    pred = model.predict(sample)

    assert pred.shape == (5,)


def test_prediction_type():  # ensures correct output type
    sample = np.random.rand(1, n_features)
    pred = model.predict(sample)

    assert isinstance(pred, np.ndarray)


def test_no_nan_in_prediction():  # ensures model does not produce NaN values
    sample = np.random.rand(3, n_features)
    pred = model.predict(sample)

    assert not np.isnan(pred).any()


def test_model_consistency():  # checks that the model produces the same output for the same input
    sample = np.random.rand(1, n_features)

    pred1 = model.predict(sample)
    pred2 = model.predict(sample)

    assert np.array_equal(pred1, pred2)


def test_invalid_input_shape():  # verifies that the model raises an error for incorrect input shapes
    wrong_sample = np.random.rand(1, n_features + 5)

    with pytest.raises(ValueError):
        model.predict(wrong_sample)