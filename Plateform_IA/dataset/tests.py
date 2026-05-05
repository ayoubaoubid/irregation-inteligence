import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse


class DatasetPagesTests(TestCase):
    def test_dataset_pages_render_successfully(self):
        page_names = ["dataset", "analysis"]

        for page_name in page_names:
            with self.subTest(page=page_name):
                response = self.client.get(reverse(page_name))
                self.assertEqual(response.status_code, 200)


class AddDataValidationTests(TestCase):
    def setUp(self):
        self.valid_payload = {
            "Soil_pH": "6.5",
            "Soil_Moisture": "45.2",
            "Organic_Carbon": "1.2",
            "Electrical_Conductivity": "2.1",
            "Temperature_C": "25.5",
            "Humidity": "60.0",
            "Rainfall_mm": "15.0",
            "Sunlight_Hours": "8.5",
            "Wind_Speed_kmh": "12.0",
            "Crop_Growth_Stage": "Vegetative",
            "Irrigation_Type": "Drip",
            "Field_Area_hectare": "5.0",
            "Mulching_Used": "Yes",
            "Previous_Irrigation_mm": "20.0",
            "Irrigation_Need": "Low",
        }

    def test_add_data_rejects_invalid_target_choice(self):
        invalid_payload = self.valid_payload | {"Irrigation_Need": "Critical"}

        response = self.client.post(reverse("add_data"), invalid_payload)

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "Le formulaire contient des erreurs.", status_code=400)

    def test_add_data_rejects_invalid_numeric_value(self):
        invalid_payload = self.valid_payload | {"Soil_pH": "not-a-number"}

        response = self.client.post(reverse("add_data"), invalid_payload)

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "Le formulaire contient des erreurs.", status_code=400)

    def test_add_data_writes_valid_row_to_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "irrigation_prediction_Variables_Important.csv"

            with patch("dataset.views.get_dataset_csv_path", return_value=csv_path), patch(
                "dataset.views.append_new_data_for_drift"
            ) as append_mock, patch(
                "dataset.views.trigger_dvc_pipeline_async", return_value=True
            ) as pipeline_mock:
                response = self.client.post(reverse("add_data"), self.valid_payload)

            self.assertEqual(response.status_code, 302)
            self.assertTrue(csv_path.exists())
            append_mock.assert_called_once()
            pipeline_mock.assert_called_once()

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Irrigation_Need"], "Low")
            self.assertEqual(rows[0]["Irrigation_Type"], "Drip")
