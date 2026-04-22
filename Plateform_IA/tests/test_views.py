"""
Tests de chargement des vues (pages) et de soumission de formulaires.

Tous les tests utilisent le client Django en mémoire — aucune base de données,
aucun serveur externe requis.
"""

import pytest
from unittest.mock import patch, mock_open


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
#  DATASET — Pages et formulaire d'ajout de données
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.views
class TestDatasetPages:
    """Vérifie les pages du module dataset."""

    @pytest.mark.parametrize("url, expected_template", [
        ("/dataset/", "dataset/dataset.html"),
        ("/dataset/analysis/", "dataset/analysis.html"),
        ("/dataset/add-data/", "dataset/add_data.html"),
    ])
    def test_page_loads_200(self, client, url, expected_template):
        """GET sur chaque URL dataset → 200 + bon template."""
        response = client.get(url)
        assert response.status_code == 200
        assert expected_template in [t.name for t in response.templates]


@pytest.mark.forms
class TestAddDataForm:
    """Vérifie la soumission du formulaire d'ajout de données."""

    @patch("builtins.open", mock_open())
    def test_add_data_success(self, client, valid_add_data_form):
        """POST valide → 200 + affichage du message de succès."""
        response = client.post("/dataset/add-data/", data=valid_add_data_form)
        assert response.status_code == 200
        assert response.context['success'] is True

    @patch("builtins.open", mock_open())
    def test_add_data_missing_fields(self, client):
        """POST avec champs manquants → la vue ne plante pas (200)."""
        response = client.post("/dataset/add-data/", data={})
        assert response.status_code == 200


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
#  PAGE 404
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.views
class TestErrorPages:
    """Vérifie le comportement sur des URLs inexistantes."""

    def test_404_on_unknown_url(self, client):
        """GET sur une URL inexistante → 404."""
        response = client.get("/cette-page-nexiste-pas/")
        assert response.status_code == 404
