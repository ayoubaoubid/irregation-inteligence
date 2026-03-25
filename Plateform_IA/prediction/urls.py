from django.urls import path
from . import views

urlpatterns = [
    path('model/', views.ml_model, name='ml_model'),
    path('form/', views.prediction, name='prediction'),
]
