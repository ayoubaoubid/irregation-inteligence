"""
Fixtures partagées pour tous les tests pytest.

- `client` : client HTTP Django (sans base de données).
- `valid_prediction_data` : données de formulaire valides pour la prédiction.

Adapté au code actuel : la vue prediction utilise une logique locale
avec les champs : temperature, humidity, soil_moisture, rainfall, wind.
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
        'temperature': '25.0',
        'humidity': '60.0',
        'soil_moisture': '30.0',
        'rainfall': '0.0',
        'wind': '10.0',
    }


@pytest.fixture
def hot_dry_data():
    """Données simulant des conditions chaudes et sèches (irrigation nécessaire)."""
    return {
        'temperature': '45.0',
        'humidity': '10.0',
        'soil_moisture': '5.0',
        'rainfall': '0.0',
        'wind': '25.0',
    }


@pytest.fixture
def wet_cool_data():
    """Données simulant des conditions humides et fraîches (pas d'irrigation)."""
    return {
        'temperature': '15.0',
        'humidity': '90.0',
        'soil_moisture': '80.0',
        'rainfall': '50.0',
        'wind': '5.0',
    }
