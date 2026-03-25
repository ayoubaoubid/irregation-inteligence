from django.shortcuts import render

def dashboard(request):
    return render(request, 'dashboard/dashboard.html')

def field_monitoring(request):
    return render(request, 'dashboard/field.html')
