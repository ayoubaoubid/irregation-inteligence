"""
Tests de chargement des vues (pages) et de soumission de formulaires.

Tous les tests utilisent le client Django en mémoire — aucune base de données,
aucun serveur externe requis.

Adapté au code actuel :
- core : 5 pages statiques
- dataset : 2 pages (dataset + analysis), pas de formulaire add-data
- prediction : page modèle + formulaire de prédiction (logique locale)
"""

import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  CORE — Pages statiques
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.views
class TestCorePages:
    """Vérifie que chaque page du module core se charge correctement (HTTP 200)."""

    @pytest.mark.parametrize("url, expected_template", [
        ("/", "core/home.html"),
        ("/problem/", "core/problem.html"),
        ("/architecture/", "core/architecture.html"),
        ("/impact/", "core/impact.html"),
        ("/about/", "core/about.html"),
    ])
    def test_page_loads_200(self, client, url, expected_template):
        """GET sur chaque URL core → 200 + bon template."""
        response = client.get(url)
        assert response.status_code == 200
        assert expected_template in [t.name for t in response.templates]


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET — Pages statiques
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.views
class TestDatasetPages:
    """Vérifie les pages du module dataset."""

    @pytest.mark.parametrize("url, expected_template", [
        ("/dataset/", "dataset/dataset.html"),
        ("/dataset/analysis/", "dataset/analysis.html"),
    ])
    def test_page_loads_200(self, client, url, expected_template):
        """GET sur chaque URL dataset → 200 + bon template."""
        response = client.get(url)
        assert response.status_code == 200
        assert expected_template in [t.name for t in response.templates]


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION — Pages
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.views
class TestPredictionPages:
    """Vérifie les pages du module prediction."""

    @pytest.mark.parametrize("url, expected_template", [
        ("/prediction/model/", "prediction/ml_model.html"),
        ("/prediction/form/", "prediction/prediction.html"),
    ])
    def test_page_loads_200(self, client, url, expected_template):
        """GET sur chaque URL prediction → 200 + bon template."""
        response = client.get(url)
        assert response.status_code == 200
        assert expected_template in [t.name for t in response.templates]


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION — Soumission du formulaire
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.forms
class TestPredictionForm:
    """Vérifie la soumission du formulaire de prédiction (logique locale)."""

    def test_valid_prediction_returns_result(self, client, valid_prediction_data):
        """POST valide → 200 + résultat affiché."""
        response = client.post("/prediction/form/", data=valid_prediction_data)
        assert response.status_code == 200
        assert response.context['result'] is True
        assert 'needs_irrigation' in response.context
        assert 'probability' in response.context
        assert 'recommendation' in response.context

    def test_prediction_preserves_inputs(self, client, valid_prediction_data):
        """Les valeurs saisies sont renvoyées dans le contexte."""
        response = client.post("/prediction/form/", data=valid_prediction_data)
        assert response.status_code == 200
        inputs = response.context['inputs']
        assert inputs['temperature'] == 25.0
        assert inputs['humidity'] == 60.0
        assert inputs['soil_moisture'] == 30.0

    def test_prediction_irrigation_needed(self, client):
        """Conditions extrêmes (chaud, sec) → irrigation recommandée."""
        data = {
            'temperature': '45.0',
            'humidity': '10.0',
            'soil_moisture': '5.0',
            'rainfall': '0.0',
            'wind': '25.0',
        }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['needs_irrigation'] is True
        assert 'Recommended' in response.context['recommendation']

    def test_prediction_no_irrigation_needed(self, client):
        """Conditions humides → pas d'irrigation."""
        data = {
            'temperature': '15.0',
            'humidity': '90.0',
            'soil_moisture': '80.0',
            'rainfall': '50.0',
            'wind': '5.0',
        }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['needs_irrigation'] is False
        assert 'No Irrigation' in response.context['recommendation']


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION — Données invalides
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.errors
class TestPredictionInvalidInput:
    """L'utilisateur envoie des données malformées."""

    def test_non_numeric_value(self, client):
        """Valeur non-numérique → erreur gérée proprement."""
        bad_data = {
            'temperature': 'pas-un-nombre',
            'humidity': '50',
            'soil_moisture': '30',
            'rainfall': '0',
            'wind': '10',
        }
        response = client.post("/prediction/form/", data=bad_data)
        assert response.status_code == 200
        assert 'error' in response.context
        assert 'Invalid' in response.context['error']

    def test_empty_form_uses_defaults(self, client):
        """Formulaire vide → les valeurs par défaut sont utilisées, pas de crash."""
        response = client.post("/prediction/form/", data={})
        assert response.status_code == 200
        # La vue utilise des défauts (25, 50, 30, 0, 10) → pas de crash
        assert response.context['result'] is True


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 404
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.views
class TestErrorPages:
    """Vérifie le comportement sur des URLs inexistantes."""

    def test_404_on_unknown_url(self, client):
        """GET sur une URL inexistante → 404."""
        response = client.get("/cette-page-nexiste-pas/")
        assert response.status_code == 404
