from django.urls import path
from . import views

urlpatterns = [
    path('', views.dataset, name='dataset'),
    path('analysis/', views.analysis, name='analysis'),
    path('add-data/', views.add_data, name='add_data'),
]

