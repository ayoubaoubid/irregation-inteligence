import csv
from django.shortcuts import render
from django.conf import settings

def dataset(request):
    return render(request, 'dataset/dataset.html')

def analysis(request):
    return render(request, 'dataset/analysis.html')

def add_data(request):
    if request.method == 'POST':
        data = [
            request.POST.get('Soil_pH'),
            request.POST.get('Soil_Moisture'),
            request.POST.get('Organic_Carbon'),
            request.POST.get('Electrical_Conductivity'),
            request.POST.get('Temperature_C'),
            request.POST.get('Humidity'),
            request.POST.get('Rainfall_mm'),
            request.POST.get('Sunlight_Hours'),
            request.POST.get('Wind_Speed_kmh'),
            request.POST.get('Crop_Growth_Stage'),
            request.POST.get('Irrigation_Type'),
            request.POST.get('Field_Area_hectare'),
            request.POST.get('Mulching_Used'),
            request.POST.get('Previous_Irrigation_mm'),
            request.POST.get('Irrigation_Need')
        ]

        # Keep dataset ingestion project-relative so local runs and containers use the same path.
        csv_path = settings.BASE_DIR.parent / 'DataOps' / 'Statics' / 'irrigation_prediction_Variables_Important.csv'
        with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            
        return render(request, 'dataset/add_data.html', {'success': True})
        
    return render(request, 'dataset/add_data.html')

def add_data(request):
    if request.method == 'POST':
        data = [
            request.POST.get('Soil_pH'),
            request.POST.get('Soil_Moisture'),
            request.POST.get('Organic_Carbon'),
            request.POST.get('Electrical_Conductivity'),
            request.POST.get('Temperature_C'),
            request.POST.get('Humidity'),
            request.POST.get('Rainfall_mm'),
            request.POST.get('Sunlight_Hours'),
            request.POST.get('Wind_Speed_kmh'),
            request.POST.get('Crop_Growth_Stage'),
            request.POST.get('Irrigation_Type'),
            request.POST.get('Field_Area_hectare'),
            request.POST.get('Mulching_Used'),
            request.POST.get('Previous_Irrigation_mm'),
            request.POST.get('Irrigation_Need')
        ]
        
        # Adding to the static CSV
        csv_path = r'd:/MLOPS_env/irregation-inteligence/DataOps/Statics/irrigation_prediction_Variables_Important.csv'
        with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            
        return render(request, 'dataset/add_data.html', {'success': True})
        
    return render(request, 'dataset/add_data.html')
