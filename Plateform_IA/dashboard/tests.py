from django.test import TestCase
from django.urls import reverse


class DashboardPagesTests(TestCase):
    def test_dashboard_pages_render_successfully(self):
        page_names = ["dashboard", "field_monitoring"]

        for page_name in page_names:
            with self.subTest(page=page_name):
                response = self.client.get(reverse(page_name))
                self.assertEqual(response.status_code, 200)
