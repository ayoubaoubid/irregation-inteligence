from django.shortcuts import render

def home(request):
    return render(request, 'core/home.html')

def problem(request):
    return render(request, 'core/problem.html')

def architecture(request):
    return render(request, 'core/architecture.html')

def impact(request):
    return render(request, 'core/impact.html')

def about(request):
    return render(request, 'core/about.html')
