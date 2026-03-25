from django.shortcuts import render

def dataset(request):
    return render(request, 'dataset/dataset.html')

def analysis(request):
    return render(request, 'dataset/analysis.html')
