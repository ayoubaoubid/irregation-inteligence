"""
Tests d'intégration : interaction entre la vue prediction et l'API modèle.

L'API externe (http://model-api:5000/predict) est systématiquement mockée
via `requests_mock` pour garantir des tests rapides et reproductibles.
"""

import pytest
import requests_mock as rm


API_URL = "http://model-api:5000/predict"


# ═══════════════════════════════════════════════════════════════════════════
#  PRÉDICTION — API modèle OK
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPredictionAPISuccess:
    """Cas nominal : l'API modèle répond 200 avec un résultat valide."""

    def test_prediction_returns_result(
        self, client, valid_prediction_data, api_success_response
    ):
        """POST formulaire → appel API mocké → résultat affiché."""
        with rm.Mocker() as m:
            m.post(API_URL, json=api_success_response, status_code=200)

            response = client.post(
                "/prediction/form/", data=valid_prediction_data
            )

        assert response.status_code == 200
        # Le contexte doit contenir le résultat de l'API
        assert 'result' in response.context
        assert response.context['result'] == api_success_response
        # Pas d'erreur
        assert response.context.get('error') is None

    def test_prediction_preserves_inputs(
        self, client, valid_prediction_data, api_success_response
    ):
        """Les valeurs saisies sont renvoyées dans le contexte (pour ré-affichage)."""
        with rm.Mocker() as m:
            m.post(API_URL, json=api_success_response, status_code=200)

            response = client.post(
                "/prediction/form/", data=valid_prediction_data
            )

        assert response.status_code == 200
        inputs = response.context['inputs']
        assert inputs['soil_ph'] == '6.5'
        assert inputs['crop_growth_stage'] == 'Vegetative'
        assert inputs['irrigation_type'] == 'Rainfed'


# ═══════════════════════════════════════════════════════════════════════════
#  PRÉDICTION — API modèle en erreur
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.errors
class TestPredictionAPIError:
    """L'API modèle retourne un code d'erreur HTTP (ex: 500)."""

    def test_api_returns_500(
        self, client, valid_prediction_data, api_error_response
    ):
        """API → 500 : la page s'affiche avec un message d'erreur."""
        with rm.Mocker() as m:
            m.post(API_URL, json=api_error_response, status_code=500)

            response = client.post(
                "/prediction/form/", data=valid_prediction_data
            )

        assert response.status_code == 200  # la page Django se charge
        assert 'error' in response.context
        assert 'Erreur API (500)' in response.context['error']

    def test_api_returns_400(self, client, valid_prediction_data):
        """API → 400 : la page s'affiche avec un message d'erreur."""
        with rm.Mocker() as m:
            m.post(API_URL, json={'error': 'Bad Request'}, status_code=400)

            response = client.post(
                "/prediction/form/", data=valid_prediction_data
            )

        assert response.status_code == 200
        assert 'error' in response.context
        assert '400' in response.context['error']


# ═══════════════════════════════════════════════════════════════════════════
#  PRÉDICTION — API inaccessible (timeout / connexion refusée)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.errors
class TestPredictionAPIDown:
    """L'API modèle est complètement inaccessible."""

    def test_api_connection_error(self, client, valid_prediction_data):
        """Connexion refusée → message d'erreur clair, pas de crash."""
        import requests

        with rm.Mocker() as m:
            m.post(API_URL, exc=requests.exceptions.ConnectionError(
                "Connection refused"
            ))

            response = client.post(
                "/prediction/form/", data=valid_prediction_data
            )

        assert response.status_code == 200
        assert 'error' in response.context
        assert 'Impossible de se connecter' in response.context['error']

    def test_api_timeout(self, client, valid_prediction_data):
        """Timeout → message d'erreur clair, pas de crash."""
        import requests

        with rm.Mocker() as m:
            m.post(API_URL, exc=requests.exceptions.Timeout(
                "Request timed out"
            ))

            response = client.post(
                "/prediction/form/", data=valid_prediction_data
            )

        assert response.status_code == 200
        assert 'error' in response.context
        assert 'Impossible de se connecter' in response.context['error']


# ═══════════════════════════════════════════════════════════════════════════
#  PRÉDICTION — Données invalides envoyées par l'utilisateur
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.errors
class TestPredictionInvalidInput:
    """L'utilisateur envoie des données malformées."""

    def test_non_numeric_value(self, client, valid_prediction_data):
        """Une valeur non-numérique dans un champ numérique → erreur ValueError."""
        bad_data = valid_prediction_data.copy()
        bad_data['soil_ph'] = 'pas-un-nombre'

        response = client.post("/prediction/form/", data=bad_data)

        assert response.status_code == 200
        assert 'error' in response.context
        assert 'invalide' in response.context['error'].lower()

    def test_empty_form_submission(self, client):
        """Formulaire vide → erreur gérée proprement."""
        response = client.post("/prediction/form/", data={})

        assert response.status_code == 200
        # Avec des valeurs par défaut dans la vue, ça peut réussir
        # ou échouer proprement — le serveur ne doit pas crasher (pas de 500)


# ═══════════════════════════════════════════════════════════════════════════
#  PRÉDICTION — Différentes valeurs catégorielles (one-hot encoding)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPredictionOneHotEncoding:
    """Vérifie que l'encodage one-hot est correct pour les champs catégoriels."""

    @pytest.mark.parametrize("crop_stage", [
        'Flowering', 'Harvest', 'Sowing', 'Vegetative',
    ])
    def test_crop_growth_stages(
        self, client, valid_prediction_data, api_success_response, crop_stage
    ):
        """Chaque valeur de Crop_Growth_Stage est acceptée."""
        data = valid_prediction_data.copy()
        data['crop_growth_stage'] = crop_stage

        with rm.Mocker() as m:
            adapter = m.post(API_URL, json=api_success_response, status_code=200)

            response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        # Vérifier que le payload envoyé à l'API a le bon one-hot
        sent_payload = adapter.last_request.json()
        assert sent_payload[f'Crop_Growth_Stage_{crop_stage}'] == 1.0

    @pytest.mark.parametrize("irrig_type", [
        'Canal', 'Drip', 'Rainfed', 'Sprinkler',
    ])
    def test_irrigation_types(
        self, client, valid_prediction_data, api_success_response, irrig_type
    ):
        """Chaque valeur de Irrigation_Type est acceptée."""
        data = valid_prediction_data.copy()
        data['irrigation_type'] = irrig_type

        with rm.Mocker() as m:
            adapter = m.post(API_URL, json=api_success_response, status_code=200)

            response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        sent_payload = adapter.last_request.json()
        assert sent_payload[f'Irrigation_Type_{irrig_type}'] == 1.0

    @pytest.mark.parametrize("mulching", ['Yes', 'No'])
    def test_mulching_values(
        self, client, valid_prediction_data, api_success_response, mulching
    ):
        """Les deux valeurs de Mulching_Used sont acceptées."""
        data = valid_prediction_data.copy()
        data['mulching_used'] = mulching

        with rm.Mocker() as m:
            adapter = m.post(API_URL, json=api_success_response, status_code=200)

            response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        sent_payload = adapter.last_request.json()
        assert sent_payload[f'Mulching_Used_{mulching}'] == 1.0
