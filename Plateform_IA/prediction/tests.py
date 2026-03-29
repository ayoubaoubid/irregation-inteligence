from django.test import TestCase
from django.urls import reverse


class PredictionPagesTests(TestCase):
    def test_prediction_pages_render_successfully(self):
        page_names = ["ml_model", "prediction"]

        for page_name in page_names:
            with self.subTest(page=page_name):
                response = self.client.get(reverse(page_name))
                self.assertEqual(response.status_code, 200)

    def test_prediction_form_returns_result_context(self):
        response = self.client.post(
            reverse("prediction"),
            {
                "temperature": 34,
                "humidity": 20,
                "soil_moisture": 15,
                "rainfall": 0,
                "wind": 12,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["result"])
        self.assertIn("recommendation", response.context)
