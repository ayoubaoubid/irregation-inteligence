"""
Fixtures partagées pour tous les tests pytest.
"""

import pytest
from django.test import Client

@pytest.fixture
def client():
    return Client()

@pytest.fixture
def valid_prediction_data():
    return {
        'soil_ph': '6.5',
        'soil_moisture': '30.0',
        'organic_carbon': '1.0',
        'electrical_conductivity': '1.5',
        'temperature_c': '25.0',
        'humidity': '60.0',
        'rainfall_mm': '0.0',
        'sunlight_hours': '8.0',
        'wind_speed_kmh': '10.0',
        'crop_growth_stage': 'Vegetative',
        'irrigation_type': 'Drip',
        'field_area_hectare': '1.0',
        'mulching_used': 'No',
        'previous_irrigation_mm': '0.0'
    }

@pytest.fixture
def hot_dry_data():
    return {
        'temperature_c': '45.0',
        'humidity': '10.0',
        'soil_moisture': '5.0',
        'rainfall_mm': '0.0',
        'wind_speed_kmh': '25.0',
    }

@pytest.fixture
def wet_cool_data():
    return {
        'temperature_c': '15.0',
        'humidity': '90.0',
        'soil_moisture': '80.0',
        'rainfall_mm': '50.0',
        'wind_speed_kmh': '5.0',
    }

@pytest.fixture
def api_success_response():
    return {
        "prediction_label": "High",
        "confidence": 85.5,
        "probabilities": {"High": 85.5, "Medium": 10.0, "Low": 4.5},
        "prediction_id": "test-pred-123"
    }

@pytest.fixture
def api_low_response():
    return {
        "prediction_label": "Low",
        "confidence": 90.0,
        "probabilities": {"High": 2.0, "Medium": 8.0, "Low": 90.0},
        "prediction_id": "test-pred-456"
    }
