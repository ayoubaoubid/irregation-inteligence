"""
Fixtures partagées pour tous les tests pytest.

- `client` : client HTTP Django (sans base de données).
- `valid_prediction_data` : données de formulaire valides pour la prédiction.
- `api_success_response` : réponse JSON simulée d'une API modèle réussie.
- `mock_api_success` / `mock_api_error` / `mock_api_timeout` : mocks requests.
"""

import pytest
from django.test import Client


# ── Django test client (pas de DB nécessaire) ──────────────────────────────

@pytest.fixture
def client():
    """Retourne un client Django pour simuler des requêtes HTTP."""
    return Client()


# ── Données de formulaire valides ──────────────────────────────────────────

@pytest.fixture
def valid_prediction_data():
    """Données POST réalistes pour le formulaire de prédiction."""
    return {
        'soil_ph': '6.5',
        'soil_moisture': '30.0',
        'organic_carbon': '0.8',
        'electrical_conductivity': '1.5',
        'temperature_c': '25.0',
        'humidity': '60.0',
        'rainfall_mm': '0.0',
        'sunlight_hours': '8.0',
        'wind_speed_kmh': '10.0',
        'crop_growth_stage': 'Vegetative',
        'irrigation_type': 'Rainfed',
        'field_area_hectare': '1.0',
        'mulching_used': 'Yes',
        'previous_irrigation_mm': '0.0',
    }


@pytest.fixture
def valid_add_data_form():
    """Données POST réalistes pour le formulaire d'ajout de données."""
    return {
        'Soil_pH': '6.5',
        'Soil_Moisture': '30.0',
        'Organic_Carbon': '0.8',
        'Electrical_Conductivity': '1.5',
        'Temperature_C': '25.0',
        'Humidity': '60.0',
        'Rainfall_mm': '0.0',
        'Sunlight_Hours': '8.0',
        'Wind_Speed_kmh': '10.0',
        'Crop_Growth_Stage': 'Vegetative',
        'Irrigation_Type': 'Rainfed',
        'Field_Area_hectare': '1.0',
        'Mulching_Used': 'Yes',
        'Previous_Irrigation_mm': '0.0',
        'Irrigation_Need': 'Yes',
    }


# ── Réponses API simulées ─────────────────────────────────────────────────

@pytest.fixture
def api_success_response():
    """Réponse JSON attendue quand l'API modèle fonctionne."""
    return {
        'prediction': 'High',
        'probability': 0.87,
        'message': 'Irrigation fortement recommandée',
    }


@pytest.fixture
def api_error_response():
    """Réponse JSON renvoyée quand l'API modèle retourne une erreur 500."""
    return {
        'error': 'Internal Server Error',
    }
