from django.shortcuts import render, redirect
from .models import PlantingInstance, PlantingParameters


def save_parameters(request):
    if request.method == 'POST':
        instance_id = request.POST.get('instance')
        instance_name = request.POST.get('instance_name')
        instance_author = request.POST.get('instance_author')

        instance, created = PlantingInstance.objects.get_or_create(
            name=instance_name,
            defaults={'author': instance_author}
        )

        PlantingParameters.objects.filter(instance=instance).delete()

        p_values = request.POST.getlist('p')
        k_values = request.POST.getlist('k')
        temperature_values = request.POST.getlist('temperature')
        rainfall_values = request.POST.getlist('rainfall')
        humidity_values = request.POST.getlist('humidity')
        ph_values = request.POST.getlist('ph')

        for p, k, temperature, rainfall, humidity, ph in zip(
            p_values, k_values, temperature_values, rainfall_values, humidity_values, ph_values
        ):
            if p and k and temperature and rainfall and humidity and ph:  
                try:
                    parameters = PlantingParameters(
                        instance=instance,
                        p=float(p),
                        k=float(k),
                        temperature=float(temperature),
                        rainfall=float(rainfall),
                        humidity=float(humidity),
                        ph=float(ph)
                    )
                    parameters.save()
                except ValueError:
                    pass

    return redirect('algorithm')

def algorithm(request):
    instances = PlantingInstance.objects.all()
    return render(request, "algorithm/algorithm.html", {'instances': instances})

