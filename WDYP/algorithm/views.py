from django.shortcuts import render, redirect, get_object_or_404
from .models import PlantingInstance, PlantingParameters
from django.http import HttpResponse
import joblib
import numpy as np
from django.contrib import messages
import os
from django.conf import settings
import pandas as pd
from django.shortcuts import render
from sklearn.preprocessing import MinMaxScaler

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
        
        n_values = request.POST.getlist('n')
        p_values = request.POST.getlist('p')
        k_values = request.POST.getlist('k')
        temperature_values = request.POST.getlist('temperature')
        humidity_values = request.POST.getlist('humidity')
        ph_values = request.POST.getlist('ph')
        rainfall_values = request.POST.getlist('rainfall')
        for n, p, k, temperature, humidity, ph, rainfall in zip(
            n_values, p_values, k_values, temperature_values,
            humidity_values, ph_values, rainfall_values,
        ):
            if n and p and k and temperature and humidity and ph and rainfall: 
                try:
                    parameters = PlantingParameters(
                        instance=instance,
                        n=float(n),
                        p=float(p),
                        k=float(k),
                        temperature=float(temperature),
                        humidity=float(humidity),
                        ph=float(ph),
                        rainfall=float(rainfall)
                    )
                    parameters.save()
                except ValueError:
                    pass
        print("POST data: ", request.POST)
        if 'execute_algorithm' in request.POST:
            print("Executing algorithm...")
            average_n = sum([float(n) for n in n_values]) / len(n_values)
            average_p = sum([float(p) for p in p_values]) / len(p_values)
            average_k = sum([float(k) for k in k_values]) / len(k_values)
            average_temperature = sum([float(temperature) for temperature in temperature_values]) / len(temperature_values)
            average_humidity = sum([float(humidity) for humidity in humidity_values]) / len(humidity_values)
            average_ph = sum([float(ph) for ph in ph_values]) / len(ph_values)
            average_rainfall = sum([float(rainfall) for rainfall in rainfall_values]) / len(rainfall_values)
            
            average_values = [
                average_n, average_p, average_k, average_temperature, average_humidity, average_ph, average_rainfall
            ]
            
           
            model_path = os.path.join(settings.BASE_DIR,"train_algorithm/trained_models")  # Use settings.model_path

            csv_path = os.path.join(settings.BASE_DIR,"train_algorithm/Crop_recommendation.csv")  # Use settings.csv_path

            df = pd.read_csv(csv_path)
            X = df.drop('label', axis=1)

            scaler = MinMaxScaler()
            scaler.fit(X)
            print("Average values: ", average_values)
            average_values_reshaped = [average_values]
            normalized_average_values = scaler.transform(average_values_reshaped)
            

            print("Normalized average values: ", normalized_average_values)
            # Load all models from the specified directory
            model_files = os.listdir(model_path)
            models = [joblib.load(os.path.join(model_path, model_file)) for model_file in model_files]

            predictions_info = []

            for idx, loaded_model in enumerate(models):
                prediction = loaded_model.predict(np.array(normalized_average_values))
                algorithm_name = loaded_model.__class__.__name__ 
                try:
                    confidence = loaded_model.predict_proba(np.array(normalized_average_values))
                    print("Confidence: ", confidence)
                    max_confidence = np.max(confidence)
                except AttributeError:
                    max_confidence = "N/A"

                predictions_info.append({
                    'model_number': idx + 1,
                    'prediction': prediction,
                    'max_confidence': max_confidence,
                    'algorithm_name': algorithm_name
                })
            
            print(predictions_info)
            return render(request, "algorithm/algorithm.html", {'predictions_info': predictions_info, 'instance': instance})
    return redirect(f'/load_instance/?instance_id={instance.id}')
    
    

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
        messages.error(request, 'Instance deleted successfully.')
        return redirect('algorithm')
    
    messages.success(request, 'Parameters loaded successfully.')
    return render(request, "algorithm/algorithm.html", {'instances': instances,
    'instance': instance, 'parameters': parameters})


