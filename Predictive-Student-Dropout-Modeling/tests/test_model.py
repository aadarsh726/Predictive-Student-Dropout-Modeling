import joblib
import os
import pytest
import numpy as np

def test_model_files_exist():
    assert os.path.exists('models/model.pkl')
    assert os.path.exists('models/scaler.pkl')
    assert os.path.exists('models/feature_names.pkl')

def test_model_loading():
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    assert model is not None
    assert scaler is not None

def test_prediction_shape():
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Create random input vector matching scaler input shape
    # We need to know input dimension.
    # Scaler.mean_ has shape (n_features,)
    n_features = scaler.mean_.shape[0]
    
    input_data = np.random.rand(1, n_features)
    scaled_data = scaler.transform(input_data)
    
    prediction = model.predict(scaled_data)
    proba = model.predict_proba(scaled_data)
    
    assert len(prediction) == 1
    assert proba.shape == (1, 2) # Binary classification
