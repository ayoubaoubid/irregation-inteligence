from django.test import TestCase
from django.urls import reverse


class DatasetPagesTests(TestCase):
    def test_dataset_pages_render_successfully(self):
        page_names = ["dataset", "analysis"]

        for page_name in page_names:
            with self.subTest(page=page_name):
                response = self.client.get(reverse(page_name))
                self.assertEqual(response.status_code, 200)
