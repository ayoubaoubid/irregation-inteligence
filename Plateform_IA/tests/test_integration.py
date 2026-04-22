"""
Tests d'intégration : cycle complet formulaire → logique de prédiction → réponse.

La vue prediction utilise actuellement une heuristique locale :
  score = (temperature * 0.5) - (humidity * 0.2) - (soil_moisture * 0.8) - (rainfall * 1.5)
  needs_irrigation = score > 0
  probability = min(100, max(0, int(score + 50)))

Ces tests valident le comportement de bout en bout (POST → calcul → template).
"""

import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  CYCLE COMPLET — Conditions normales
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPredictionFullCycle:
    """Teste le cycle complet : formulaire → logique → réponse."""

    def test_normal_conditions_returns_result(self, client, valid_prediction_data):
        """POST avec données normales → réponse 200 + contexte complet."""
        response = client.post("/prediction/form/", data=valid_prediction_data)

        assert response.status_code == 200
        ctx = response.context
        assert ctx['result'] is True
        assert isinstance(ctx['needs_irrigation'], bool)
        assert isinstance(ctx['probability'], int)
        assert 0 <= ctx['probability'] <= 100
        assert isinstance(ctx['recommendation'], str)
        assert len(ctx['recommendation']) > 0

    def test_inputs_are_preserved_as_floats(self, client, valid_prediction_data):
        """Les inputs sont convertis en float et renvoyés dans le contexte."""
        response = client.post("/prediction/form/", data=valid_prediction_data)

        assert response.status_code == 200
        inputs = response.context['inputs']
        assert inputs['temperature'] == 25.0
        assert inputs['humidity'] == 60.0
        assert inputs['soil_moisture'] == 30.0
        assert inputs['rainfall'] == 0.0
        assert inputs['wind'] == 10.0


# ═══════════════════════════════════════════════════════════════════════════
#  SCÉNARIO — Conditions chaudes et sèches → irrigation nécessaire
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIrrigationNeeded:
    """Conditions extrêmes de sécheresse → irrigation recommandée."""

    def test_hot_dry_triggers_irrigation(self, client, hot_dry_data):
        """Température élevée + faible humidité → irrigation nécessaire."""
        response = client.post("/prediction/form/", data=hot_dry_data)

        assert response.status_code == 200
        assert response.context['needs_irrigation'] is True
        assert 'Recommended' in response.context['recommendation']

    def test_hot_dry_high_probability(self, client, hot_dry_data):
        """Conditions sèches → probabilité élevée."""
        response = client.post("/prediction/form/", data=hot_dry_data)

        assert response.status_code == 200
        assert response.context['probability'] > 50


# ═══════════════════════════════════════════════════════════════════════════
#  SCÉNARIO — Conditions humides → pas d'irrigation
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestNoIrrigationNeeded:
    """Conditions humides → irrigation non nécessaire."""

    def test_wet_cool_no_irrigation(self, client, wet_cool_data):
        """Humidité forte + pluie → pas d'irrigation."""
        response = client.post("/prediction/form/", data=wet_cool_data)

        assert response.status_code == 200
        assert response.context['needs_irrigation'] is False
        assert 'No Irrigation' in response.context['recommendation']

    def test_wet_cool_low_probability(self, client, wet_cool_data):
        """Conditions humides → probabilité faible (clampée à 0 minimum)."""
        response = client.post("/prediction/form/", data=wet_cool_data)

        assert response.status_code == 200
        assert response.context['probability'] <= 50


# ═══════════════════════════════════════════════════════════════════════════
#  BOUNDARY — Valeurs limites de la probabilité
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestProbabilityBoundaries:
    """Vérifie que la probabilité est toujours clampée entre 0 et 100."""

    def test_probability_never_below_zero(self, client):
        """Même en conditions extrêmement humides, probability >= 0."""
        data = {
            'temperature': '0.0',
            'humidity': '100.0',
            'soil_moisture': '100.0',
            'rainfall': '200.0',
            'wind': '0.0',
        }
        response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        assert response.context['probability'] >= 0

    def test_probability_never_above_100(self, client):
        """Même en conditions extrêmement sèches, probability <= 100."""
        data = {
            'temperature': '60.0',
            'humidity': '0.0',
            'soil_moisture': '0.0',
            'rainfall': '0.0',
            'wind': '50.0',
        }
        response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        assert response.context['probability'] <= 100


# ═══════════════════════════════════════════════════════════════════════════
#  PARAMETRIZE — Différentes combinaisons de champs
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPredictionParametrized:
    """Tests paramétrés pour vérifier la cohérence de la heuristique."""

    @pytest.mark.parametrize("temperature,expected_irrigation", [
        ('50.0', True),     # très chaud → irrigation
        ('10.0', False),    # frais → pas d'irrigation
    ])
    def test_temperature_impact(self, client, temperature, expected_irrigation):
        """La température est le facteur principal (coefficient 0.5 positif)."""
        data = {
            'temperature': temperature,
            'humidity': '50.0',
            'soil_moisture': '20.0',
            'rainfall': '0.0',
            'wind': '10.0',
        }
        response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        assert response.context['needs_irrigation'] is expected_irrigation

    @pytest.mark.parametrize("rainfall,expected_irrigation", [
        ('0.0', True),      # pas de pluie → irrigation
        ('50.0', False),    # forte pluie → pas d'irrigation
    ])
    def test_rainfall_impact(self, client, rainfall, expected_irrigation):
        """La pluie réduit fortement le besoin d'irrigation (coefficient -1.5)."""
        data = {
            'temperature': '35.0',
            'humidity': '30.0',
            'soil_moisture': '10.0',
            'rainfall': rainfall,
            'wind': '10.0',
        }
        response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        assert response.context['needs_irrigation'] is expected_irrigation


# ═══════════════════════════════════════════════════════════════════════════
#  ERREURS — Robustesse du formulaire
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.errors
class TestPredictionErrors:
    """Vérifie la gestion d'erreurs du formulaire."""

    def test_non_numeric_temperature(self, client):
        """Valeur non-numérique → message d'erreur, pas de crash."""
        data = {
            'temperature': 'abc',
            'humidity': '50',
            'soil_moisture': '30',
            'rainfall': '0',
            'wind': '10',
        }
        response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        assert 'error' in response.context
        assert 'Invalid' in response.context['error']

    def test_all_fields_non_numeric(self, client):
        """Tous les champs invalides → message d'erreur, pas de crash."""
        data = {
            'temperature': 'hot',
            'humidity': 'wet',
            'soil_moisture': 'dry',
            'rainfall': 'none',
            'wind': 'calm',
        }
        response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        assert 'error' in response.context

    def test_empty_form_uses_defaults(self, client):
        """Formulaire vide → les valeurs par défaut fonctionnent."""
        response = client.post("/prediction/form/", data={})

        assert response.status_code == 200
        assert response.context['result'] is True

    def test_negative_values_accepted(self, client):
        """Valeurs négatives (ex: température) → pas de crash."""
        data = {
            'temperature': '-10.0',
            'humidity': '80.0',
            'soil_moisture': '50.0',
            'rainfall': '20.0',
            'wind': '5.0',
        }
        response = client.post("/prediction/form/", data=data)

        assert response.status_code == 200
        assert response.context['result'] is True
