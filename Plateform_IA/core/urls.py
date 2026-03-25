from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('problem/', views.problem, name='problem'),
    path('architecture/', views.architecture, name='architecture'),
    path('impact/', views.impact, name='impact'),
    path('about/', views.about, name='about'),
]
