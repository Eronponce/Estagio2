from django.shortcuts import render, redirect
from .models import PlantingParameters

def save_parameters(request):
    if request.method == 'POST':
        PlantingParameters.objects.all().delete()
        p_values = request.POST.getlist('p')
        k_values = request.POST.getlist('k')
        temperature_values = request.POST.getlist('temperature')
        rainfall_values = request.POST.getlist('rainfall')
        humidity_values = request.POST.getlist('humidity')
        ph_values = request.POST.getlist('ph')

        for p, k, temperature, rainfall, humidity, ph in zip(
            p_values, k_values, temperature_values, rainfall_values, humidity_values, ph_values
        ):
            if p and k and temperature and rainfall and humidity and ph:  # Verifica se os campos estão preenchidos
                try:
                    parameters = PlantingParameters(
                        p=float(p),
                        k=float(k),
                        temperature=float(temperature),
                        rainfall=float(rainfall),
                        humidity=float(humidity),
                        ph=float(ph)
                    )
                    print(parameters)
                    parameters.save()
                except ValueError:
                    print("erro")
                    pass

    return redirect('algorithm')


def algorithm(request):
    parameters = PlantingParameters.objects.all()  # Busca todos os registros do banco de dados
    return render(request, "algorithm/algorithm.html", {'parameters': parameters})