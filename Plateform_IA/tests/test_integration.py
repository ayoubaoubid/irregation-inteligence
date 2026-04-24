"""
Tests d'intégration : cycle complet formulaire → logique de prédiction → réponse (Mock API).
"""
import pytest

@pytest.mark.integration
class TestPredictionFullCycle:
    def test_normal_conditions_returns_result(self, client, valid_prediction_data, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data=valid_prediction_data)
        assert response.status_code == 200
        ctx = response.context
        assert ctx['result']['prediction_label'] == "High"
        assert 'confidence' in ctx['result']

    def test_inputs_are_preserved_as_strings(self, client, valid_prediction_data, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data=valid_prediction_data)
        assert response.status_code == 200
        inputs = response.context['inputs']
        assert inputs['temperature_c'] == '25.0'
        assert inputs['humidity'] == '60.0'

@pytest.mark.integration
class TestIrrigationNeeded:
    def test_hot_dry_triggers_irrigation(self, client, hot_dry_data, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data=hot_dry_data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

    def test_hot_dry_high_probability(self, client, hot_dry_data, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data=hot_dry_data)
        assert response.status_code == 200
        assert response.context['result']['probabilities']['High'] > 50

@pytest.mark.integration
class TestNoIrrigationNeeded:
    def test_wet_cool_no_irrigation(self, client, wet_cool_data, requests_mock, api_low_response):
        requests_mock.post("http://model-api:5000/predict", json=api_low_response)
        response = client.post("/prediction/form/", data=wet_cool_data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "Low"

    def test_wet_cool_low_probability(self, client, wet_cool_data, requests_mock, api_low_response):
        requests_mock.post("http://model-api:5000/predict", json=api_low_response)
        response = client.post("/prediction/form/", data=wet_cool_data)
        assert response.status_code == 200
        assert response.context['result']['probabilities']['High'] < 50

@pytest.mark.integration
class TestProbabilityBoundaries:
    def test_probability_never_below_zero(self, client, requests_mock, api_low_response):
        requests_mock.post("http://model-api:5000/predict", json=api_low_response)
        data = { 'temperature_c': '0.0', 'humidity': '100.0', 'soil_moisture': '100.0', 'rainfall_mm': '200.0', 'wind_speed_kmh': '0.0' }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['confidence'] >= 0

    def test_probability_never_above_100(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        data = { 'temperature_c': '60.0', 'humidity': '0.0', 'soil_moisture': '0.0', 'rainfall_mm': '0.0', 'wind_speed_kmh': '50.0' }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['confidence'] <= 100

@pytest.mark.integration
class TestPredictionParametrized:
    @pytest.mark.parametrize("temperature_c,expected_label", [
        ('50.0', 'High'),
        ('10.0', 'Low'),
    ])
    def test_temperature_impact(self, client, temperature_c, expected_label, requests_mock):
        mock_resp = {"prediction_label": expected_label, "confidence": 90.0, "probabilities": {expected_label: 90.0}}
        requests_mock.post("http://model-api:5000/predict", json=mock_resp)
        data = { 'temperature_c': temperature_c, 'humidity': '50.0', 'soil_moisture': '20.0', 'rainfall_mm': '0.0', 'wind_speed_kmh': '10.0' }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == expected_label

    @pytest.mark.parametrize("rainfall_mm,expected_label", [
        ('0.0', 'High'),
        ('50.0', 'Low'),
    ])
    def test_rainfall_impact(self, client, rainfall_mm, expected_label, requests_mock):
        mock_resp = {"prediction_label": expected_label, "confidence": 90.0, "probabilities": {expected_label: 90.0}}
        requests_mock.post("http://model-api:5000/predict", json=mock_resp)
        data = { 'temperature_c': '35.0', 'humidity': '30.0', 'soil_moisture': '10.0', 'rainfall_mm': rainfall_mm, 'wind_speed_kmh': '10.0' }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == expected_label

@pytest.mark.errors
class TestPredictionErrors:
    def test_non_numeric_temperature(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        data = { 'temperature_c': 'abc', 'humidity': '50', 'soil_moisture': '30', 'rainfall_mm': '0', 'wind_speed_kmh': '10' }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        # Avec to_float qui capture l'erreur et met la valeur par defaut, 
        # on doit verifier que le fallback marche et qu'on a un resultat correct
        assert response.context['result']['prediction_label'] == "High"

    def test_all_fields_non_numeric(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        data = { 'temperature_c': 'hot', 'humidity': 'wet', 'soil_moisture': 'dry', 'rainfall_mm': 'none', 'wind_speed_kmh': 'calm' }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

    def test_empty_form_uses_defaults(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data={})
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

    def test_negative_values_accepted(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        data = { 'temperature_c': '-10.0', 'humidity': '80.0', 'soil_moisture': '50.0', 'rainfall_mm': '20.0', 'wind_speed_kmh': '5.0' }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

    def test_api_connection_error(self, client, requests_mock, valid_prediction_data):
        import requests
        requests_mock.post("http://model-api:5000/predict", exc=requests.exceptions.ConnectionError)
        response = client.post("/prediction/form/", data=valid_prediction_data)
        assert response.status_code == 200
        assert 'error' in response.context
        assert 'Impossible de se connecter' in response.context['error']