from django.shortcuts import render, redirect, get_object_or_404
from .models import PlantingInstance, PlantingParameters
from django.http import HttpResponse

def save_parameters(request):
    if request.method == 'POST':
        instance_id = request.POST.get('instance')
      
        instance_name = request.POST.get('instance_name')
        if instance_name == "":
            instance_name = "Instância sem nome"
        instance_author = request.POST.get('instance_author')
        if instance_author == "":
            instance_author = "Autor anônimo"

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


def load_instance(request):
    instance_id = request.GET.get('instance_id')
    if instance_id == "":
        return redirect('algorithm')
    instances = PlantingInstance.objects.all()
    instance = get_object_or_404(PlantingInstance, id=instance_id)
    parameters = PlantingParameters.objects.filter(instance=instance)
    # Render the template with the instance and its parameters

    if 'delete_instance' in request.GET:
        # Delete the instance and its parameters
        instance.delete()
        return redirect('algorithm')

    return render(request, "algorithm/algorithm.html", {'instances': instances,'instance': instance, 'parameters': parameters})


