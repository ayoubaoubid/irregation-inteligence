from django.shortcuts import render
from django.http import JsonResponse
import random

def ml_model(request):
    return render(request, 'prediction/ml_model.html')

def prediction(request):
    if request.method == 'POST':
        # Dummy prediction logic
        try:
            temperature = float(request.POST.get('temperature', 25))
            humidity = float(request.POST.get('humidity', 50))
            soil_moisture = float(request.POST.get('soil_moisture', 30))
            rainfall = float(request.POST.get('rainfall', 0))
            wind = float(request.POST.get('wind', 10))
            
            # Simple dummy model heuristic
            score = (temperature * 0.5) - (humidity * 0.2) - (soil_moisture * 0.8) - (rainfall * 1.5)
            
            needs_irrigation = score > 0
            probability = min(100, max(0, int((score + 50))))
            
            if needs_irrigation:
                recommendation = "Irrigation Recommended. Soil moisture is low and environmental factors indicate stress."
            else:
                recommendation = "No Irrigation Needed. Current conditions are optimal."
                
            return render(request, 'prediction/prediction.html', {
                'result': True,
                'needs_irrigation': needs_irrigation,
                'probability': probability,
                'recommendation': recommendation,
                'inputs': {
                    'temperature': temperature,
                    'humidity': humidity,
                    'soil_moisture': soil_moisture,
                    'rainfall': rainfall,
                    'wind': wind
                }
            })
        except ValueError:
            return render(request, 'prediction/prediction.html', {'error': 'Invalid input values.'})
            
    return render(request, 'prediction/prediction.html')
