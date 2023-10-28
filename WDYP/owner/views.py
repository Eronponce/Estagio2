from django.shortcuts import render

# Create your views here.
def controlOwner(request):
    return render(request, 'owner.html')

def controlProperty(request):
    return render(request, 'property.html')