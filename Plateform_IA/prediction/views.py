from django.shortcuts import render
from django.http import JsonResponse
import random

def ml_model(request):
    return render(request, 'prediction/ml_model.html')

import pandas as pd
import numpy as np

import requests

def prediction(request):
    if request.method == 'POST':
        try:
            inputs_context = {
                'soil_ph': request.POST.get('soil_ph', '6.5'),
                'soil_moisture': request.POST.get('soil_moisture', '30.0'),
                'organic_carbon': request.POST.get('organic_carbon', '0.8'),
                'electrical_conductivity': request.POST.get('electrical_conductivity', '1.5'),
                'temperature_c': request.POST.get('temperature_c', '25.0'),
                'humidity': request.POST.get('humidity', '60.0'),
                'rainfall_mm': request.POST.get('rainfall_mm', '0.0'),
                'sunlight_hours': request.POST.get('sunlight_hours', '8.0'),
                'wind_speed_kmh': request.POST.get('wind_speed_kmh', '10.0'),
                'crop_growth_stage': request.POST.get('crop_growth_stage', 'Vegetative'),
                'irrigation_type': request.POST.get('irrigation_type', 'Rainfed'),
                'field_area_hectare': request.POST.get('field_area_hectare', '1.0'),
                'mulching_used': request.POST.get('mulching_used', 'Yes'),
                'previous_irrigation_mm': request.POST.get('previous_irrigation_mm', '0.0')
            }
            
            crop_stage = inputs_context['crop_growth_stage']
            irrigation = inputs_context['irrigation_type']
            mulching = inputs_context['mulching_used']
            
            payload = {
                'Soil_pH': float(inputs_context['soil_ph']),
                'Soil_Moisture': float(inputs_context['soil_moisture']),
                'Organic_Carbon': float(inputs_context['organic_carbon']),
                'Electrical_Conductivity': float(inputs_context['electrical_conductivity']),
                'Temperature_C': float(inputs_context['temperature_c']),
                'Humidity': float(inputs_context['humidity']),
                'Rainfall_mm': float(inputs_context['rainfall_mm']),
                'Sunlight_Hours': float(inputs_context['sunlight_hours']),
                'Wind_Speed_kmh': float(inputs_context['wind_speed_kmh']),
                'Field_Area_hectare': float(inputs_context['field_area_hectare']),
                'Previous_Irrigation_mm': float(inputs_context['previous_irrigation_mm']),
                
                'Crop_Growth_Stage_Flowering': 1.0 if crop_stage == 'Flowering' else 0.0,
                'Crop_Growth_Stage_Harvest': 1.0 if crop_stage == 'Harvest' else 0.0,
                'Crop_Growth_Stage_Sowing': 1.0 if crop_stage == 'Sowing' else 0.0,
                'Crop_Growth_Stage_Vegetative': 1.0 if crop_stage == 'Vegetative' else 0.0,
                
                'Irrigation_Type_Canal': 1.0 if irrigation == 'Canal' else 0.0,
                'Irrigation_Type_Drip': 1.0 if irrigation == 'Drip' else 0.0,
                'Irrigation_Type_Rainfed': 1.0 if irrigation == 'Rainfed' else 0.0,
                'Irrigation_Type_Sprinkler': 1.0 if irrigation == 'Sprinkler' else 0.0,
                
                'Mulching_Used_No': 1.0 if mulching == 'No' else 0.0,
                'Mulching_Used_Yes': 1.0 if mulching == 'Yes' else 0.0,
            }
            
            # Appel API
            api_url = "http://model-api:5000/predict"
            try:
                response = requests.post(api_url, json=payload)
                if response.status_code == 200:
                    api_result = response.json()
                else:
                    return render(request, 'prediction/prediction.html', {
                        'error': f'Erreur API ({response.status_code}): {response.text}', 
                        'inputs': inputs_context
                    })
            except requests.exceptions.RequestException as e:
                return render(request, 'prediction/prediction.html', {
                    'error': f'Impossible de se connecter à l\'API du modèle : {str(e)}', 
                    'inputs': inputs_context
                })

            context = {
                'result': api_result,
                'inputs': inputs_context
            }
            return render(request, 'prediction/prediction.html', context)
        except ValueError as e:
            return render(request, 'prediction/prediction.html', {'error': f'Valeur d\'entrée invalide : {str(e)}'})
            
    return render(request, 'prediction/prediction.html')
