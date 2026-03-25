from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('field/', views.field_monitoring, name='field_monitoring'),
]
