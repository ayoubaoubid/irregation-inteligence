"""
Tests de chargement des vues (pages) et de soumission de formulaires.
"""

import pytest

@pytest.mark.views
class TestCorePages:
    @pytest.mark.parametrize("url, expected_template", [
        ("/", "core/home.html"),
        ("/problem/", "core/problem.html"),
        ("/architecture/", "core/architecture.html"),
        ("/impact/", "core/impact.html"),
        ("/about/", "core/about.html"),
    ])
    def test_page_loads_200(self, client, url, expected_template):
        response = client.get(url)
        assert response.status_code == 200
        assert expected_template in [t.name for t in response.templates]

@pytest.mark.views
class TestDatasetPages:
    @pytest.mark.parametrize("url, expected_template", [
        ("/dataset/", "dataset/dataset.html"),
        ("/dataset/analysis/", "dataset/analysis.html"),
    ])
    def test_page_loads_200(self, client, url, expected_template):
        response = client.get(url)
        assert response.status_code == 200
        assert expected_template in [t.name for t in response.templates]

@pytest.mark.views
class TestPredictionPages:
    @pytest.mark.parametrize("url, expected_template", [
        ("/prediction/model/", "prediction/ml_model.html"),
        ("/prediction/form/", "prediction/prediction.html"),
    ])
    def test_page_loads_200(self, client, url, expected_template):
        response = client.get(url)
        assert response.status_code == 200
        assert expected_template in [t.name for t in response.templates]

@pytest.mark.forms
class TestPredictionForm:
    def test_valid_prediction_returns_result(self, client, valid_prediction_data, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data=valid_prediction_data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

    def test_prediction_preserves_inputs(self, client, valid_prediction_data, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data=valid_prediction_data)
        assert response.status_code == 200
        inputs = response.context['inputs']
        assert inputs['temperature_c'] == '25.0'
        assert inputs['humidity'] == '60.0'

    def test_prediction_irrigation_needed(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        data = {
            'temperature_c': '45.0',
            'humidity': '10.0',
            'soil_moisture': '5.0',
            'rainfall_mm': '0.0',
            'wind_speed_kmh': '25.0',
        }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

    def test_prediction_no_irrigation_needed(self, client, requests_mock, api_low_response):
        requests_mock.post("http://model-api:5000/predict", json=api_low_response)
        data = {
            'temperature_c': '15.0',
            'humidity': '90.0',
            'soil_moisture': '80.0',
            'rainfall_mm': '50.0',
            'wind_speed_kmh': '5.0',
        }
        response = client.post("/prediction/form/", data=data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "Low"

@pytest.mark.errors
class TestPredictionInvalidInput:
    def test_non_numeric_value(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        bad_data = {
            'temperature_c': 'pas-un-nombre',
            'humidity': '50',
            'soil_moisture': '30',
            'rainfall_mm': '0',
            'wind_speed_kmh': '10',
        }
        response = client.post("/prediction/form/", data=bad_data)
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

    def test_empty_form_uses_defaults(self, client, requests_mock, api_success_response):
        requests_mock.post("http://model-api:5000/predict", json=api_success_response)
        response = client.post("/prediction/form/", data={})
        assert response.status_code == 200
        assert response.context['result']['prediction_label'] == "High"

@pytest.mark.views
class TestErrorPages:
    def test_404_on_unknown_url(self, client):
        response = client.get("/cette-page-nexiste-pas/")
        assert response.status_code == 404
