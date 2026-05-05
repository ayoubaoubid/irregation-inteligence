from django.shortcuts import render
from django.http import JsonResponse
import random


def ml_model(request):
    return render(request, 'prediction/ml_model.html')


import pandas as pd
import numpy as np

import requests
from django.conf import settings


def to_float(value, default=0.0):
    """Convert a submitted value to float and fall back when empty or invalid."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def prediction(request):
    if request.method == 'POST':
        try:
            inputs_context = {
                'soil_ph': request.POST.get('soil_ph'),
                'soil_moisture': request.POST.get('soil_moisture'),
                'organic_carbon': request.POST.get('organic_carbon'),
                'electrical_conductivity': request.POST.get('electrical_conductivity'),
                'temperature_c': request.POST.get('temperature_c'),
                'humidity': request.POST.get('humidity'),
                'rainfall_mm': request.POST.get('rainfall_mm'),
                'sunlight_hours': request.POST.get('sunlight_hours'),
                'wind_speed_kmh': request.POST.get('wind_speed_kmh'),
                'crop_growth_stage': request.POST.get('crop_growth_stage'),
                'irrigation_type': request.POST.get('irrigation_type'),
                'field_area_hectare': request.POST.get('field_area_hectare'),
                'mulching_used': request.POST.get('mulching_used'),
                'previous_irrigation_mm': request.POST.get('previous_irrigation_mm')
            }

            crop_stage = inputs_context['crop_growth_stage']
            irrigation = inputs_context['irrigation_type']
            mulching = inputs_context['mulching_used']

            payload = {
                'Soil_pH': to_float(inputs_context['soil_ph'], default=7.0),
                'Soil_Moisture': to_float(inputs_context['soil_moisture'], default=50.0),
                'Organic_Carbon': to_float(inputs_context['organic_carbon'], default=1.5),
                'Electrical_Conductivity': to_float(inputs_context['electrical_conductivity'], default=0.5),
                'Temperature_C': to_float(inputs_context['temperature_c'], default=25.0),
                'Humidity': to_float(inputs_context['humidity'], default=60.0),
                'Rainfall_mm': to_float(inputs_context['rainfall_mm'], default=10.0),
                'Sunlight_Hours': to_float(inputs_context['sunlight_hours'], default=8.0),
                'Wind_Speed_kmh': to_float(inputs_context['wind_speed_kmh'], default=10.0),
                'Field_Area_hectare': to_float(inputs_context['field_area_hectare'], default=1.0),
                'Previous_Irrigation_mm': to_float(inputs_context['previous_irrigation_mm'], default=0.0),
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

            api_url = settings.MODEL_API_PREDICT_URL

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
                    'error': f'Impossible de se connecter a l API du modele : {str(e)}',
                    'inputs': inputs_context
                })


            if 'confidence' in api_result and api_result['confidence'] <= 1:
                api_result['confidence'] = api_result['confidence'] * 100

            context = {
                'result': api_result,
                'inputs': inputs_context
            }
            return render(request, 'prediction/prediction.html', context)
        except ValueError as e:
            return render(request, 'prediction/prediction.html', {'error': f'Valeur d entree invalide : {str(e)}'})

    return render(request, 'prediction/prediction.html')
