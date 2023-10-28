from django.shortcuts import render, redirect, get_object_or_404
from .models import PlantingInstance, PlantingParameters, PredictionInfo
import joblib
import numpy as np
from django.contrib import messages
from django.conf import settings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from django.contrib.auth.decorators import login_required

@login_required
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
        
        if 'execute_algorithm' in request.POST:
            PredictionInfo.objects.filter(instance=instance).delete()

            try:
                with open(os.path.join(settings.BASE_DIR, "train_algorithm", "model_accuracies.json"), "r") as f:
                    model_accuracies = json.load(f)
            except Exception as e:
                print(f"Could not load model accuracies. Error: {e}")
                model_accuracies = {}
            print("Executing algorithm...")

            if n_values == None:
                return redirect(f'/load_instance/?instance_id={instance.id}')
            def calculate_average(values):
                if values and len(values) > 0:
                    return sum([float(val) for val in values]) / len(values)
                else:
                    return 0  # Or any other appropriate default value

            average_n = calculate_average(n_values)
            average_p = calculate_average(p_values)
            average_k = calculate_average(k_values)
            average_temperature = calculate_average(temperature_values)
            average_humidity = calculate_average(humidity_values)
            average_ph = calculate_average(ph_values)
            average_rainfall = calculate_average(rainfall_values)
            
            average_values = [
                average_n, average_p, average_k, average_temperature, average_humidity, average_ph, average_rainfall
            ]
            
           
            model_path = os.path.join(settings.BASE_DIR,"train_algorithm/trained_models")  
            csv_path = os.path.join(settings.BASE_DIR,"train_algorithm", "Crop_recommendation.csv") 
           
            df = pd.read_csv(csv_path)
            X = df.drop('label', axis=1)
           
            scaler = MinMaxScaler()
            scaler.fit(X)
            
            average_values_reshaped = [average_values]
            normalized_average_values = scaler.transform(average_values_reshaped)
            

            model_files = os.listdir(model_path)
            models = [joblib.load(os.path.join(model_path, model_file)) for model_file in model_files]

            predictions_info = []

            for idx, loaded_model in enumerate(models):
                prediction = loaded_model.predict(np.array(normalized_average_values))
                algorithm_name = loaded_model.__class__.__name__
                name_mapping = {
                    'LogisticRegression': 'Logistic Regression',
                    'RandomForestClassifier': 'Random Forest',
                    'SVC': 'SVM_RBF',
                    'DecisionTreeClassifier': 'Decision Tree',
                    'GradientBoostingClassifier': 'Gradient Boosting',
                    'KNeighborsClassifier': 'KNN_Euclidean',
                    'GaussianNB': 'Naive Bayes',
                    'LinearDiscriminantAnalysis': 'Linear Discriminant Analysis',
                    'AdaBoostClassifier': 'AdaBoost',
                    'MLPClassifier': 'MLP Neural Network'
                }

                try:
                    with open(os.path.join(settings.BASE_DIR, "train_algorithm", "model_accuracies.json"), "r") as f:
                        model_accuracies = json.load(f)
                except Exception as e:
                    print(f"Could not load model accuracies. Error: {e}")
                    model_accuracies = {}
                mapped_name = name_mapping.get(algorithm_name, algorithm_name)
                accuracy = model_accuracies.get(mapped_name, "N/A")
                try:
                    confidence = loaded_model.predict_proba(np.array(normalized_average_values))
                    max_confidence = np.max(confidence)
                except AttributeError:
                    max_confidence = "N/A"

                predictions_info.append({
                    'model_number': idx + 1,
                    'prediction': prediction,
                    'max_confidence': format(max_confidence, ".2f"),
                    'algorithm_name': algorithm_name,
                    'accuracy': format(accuracy, ".2f")  
                })
            instances = PlantingInstance.objects.all()
            parameters = PlantingParameters.objects.filter(instance=instance)
            for param in parameters:
                param.n = format(float(param.n), '.2f')
                param.p = format(float(param.p), '.2f')
                param.k = format(float(param.k), '.2f')
                param.temperature = format(float(param.temperature), '.2f')
                param.humidity = format(float(param.humidity), '.2f')
                param.ph = format(float(param.ph), '.2f')
                param.rainfall = format(float(param.rainfall), '.2f')

            sc_x = StandardScaler()
            sc_x.fit(X)

            knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=19, metric='euclidean')
            knn_classifier.fit(sc_x.transform(X), df['label'])
 
            scaled_average_values = sc_x.transform(pd.DataFrame([average_values], columns=X.columns))
            
            prediction_knn = knn_classifier.predict(scaled_average_values)
          
            culture_df = df[df['label'] == prediction_knn[0]]
            culture_df = culture_df.drop(columns=['label'])
            culture_mean = culture_df.mean()
            culture_std = culture_df.std()
            z_score = (average_values - culture_mean) / culture_std
            
            z_score_list = [str(round(val, 1)) for val in z_score]
            formatted_variation_str = ','.join(z_score_list) 
            
           
            for pred_info in predictions_info:
                PredictionInfo.objects.create(
                    instance=instance,
                    model_number=pred_info['model_number'],
                    prediction=pred_info['prediction'][0], 
                    max_confidence=float(pred_info['max_confidence']),
                    algorithm_name=pred_info['algorithm_name'],
                    accuracy=float(pred_info['accuracy']),
                    knn_variation=formatted_variation_str  # Include the variation here
    )

            print(z_score_list)
            return render(request, "algorithm/algorithm.html", {'instances': instances,'predictions_info': predictions_info, 'instance': instance ,'parameters' : parameters,'variations': z_score_list})
    return redirect(f'/load_instance/?instance_id={instance.id}')
    
    

@login_required
def algorithm(request):
    instances = PlantingInstance.objects.all()
    return render(request, "algorithm/algorithm.html", {'instances': instances})



@login_required
def load_instance(request):
    instance_id = request.GET.get('instance_id')
    if instance_id == "":
        return redirect('algorithm')
    
    instances = PlantingInstance.objects.all()
    instance = get_object_or_404(PlantingInstance, id=instance_id)
    parameters = PlantingParameters.objects.filter(instance=instance)
    for param in parameters:
        param.n = format(float(param.n), '.2f')
        param.p = format(float(param.p), '.2f')
        param.k = format(float(param.k), '.2f')
        param.temperature = format(float(param.temperature), '.2f')
        param.humidity = format(float(param.humidity), '.2f')
        param.ph = format(float(param.ph), '.2f')
        param.rainfall = format(float(param.rainfall), '.2f')
    predictions = PredictionInfo.objects.filter(instance=instance)
    for prediction in predictions:
        prediction.prediction = [prediction.prediction]

    prediction_info = PredictionInfo.objects.filter(instance=instance).first()
    formatted_variation = prediction_info.knn_variation if prediction_info else None
    try:
        formatted_variation = [prediction_info.knn_variation.split(",")]
    except AttributeError:
        formatted_variation = None
    if 'delete_instance' in request.GET:
        instance.delete()
        messages.error(request, 'Instancia e resultados deletados.')
        return redirect('algorithm')
    
    messages.success(request, 'Instancia e resultados carregados.')
    return render(request, "algorithm/algorithm.html", {
        'instances': instances,
        'instance': instance,
        'parameters': parameters,
        'predictions_info': predictions,
        'variations': formatted_variation[0] if formatted_variation and formatted_variation[0] else None

    })


@login_required
def train(request):
    
    model_path = os.path.join(settings.BASE_DIR,"train_algorithm/trained_models")
    csv_path = os.path.join(settings.BASE_DIR,"train_algorithm", "Crop_recommendation.csv")
    df = pd.read_csv(csv_path)

    x = df.drop(columns=['label'])
    y = df['label']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Random Forest', RandomForestClassifier()),
        ('SVM_RBF', SVC(probability=True)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('KNN_Euclidean', KNeighborsClassifier(n_neighbors=19, metric='euclidean')),
        ('Naive Bayes', GaussianNB()),
        ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
        ('AdaBoost', AdaBoostClassifier()),
        ('MLP Neural Network', MLPClassifier(max_iter=500))
    ]
    if os.path.exists(model_path):
       
        for filename in os.listdir(model_path):
            file_path = os.path.join(model_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


    model_accuracies = {}
    
    directory_path = os.path.join(settings.BASE_DIR,"static" ,"confusion_matrix")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
   
    def plot_simple_confusion_matrix(cm, classes, title='Confusion Matrix', image_path=''):
        try:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title(title)
            plt.tight_layout()
          
            plt.savefig(image_path)
            plt.close()
        except Exception as e:
            print(f"Failed to save image at {image_path}. Reason: {e}")

    
    for name, model in models:
        try:
            print(f"Training {name}...")
            
            model.fit(x_train, y_train)
            
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_accuracies[name] = accuracy 
            
            print(f"Accuracy of {name}: {accuracy * 100:.2f}%")
            cm = confusion_matrix(y_test, y_pred)
            model_path = os.path.join(settings.BASE_DIR, "train_algorithm", "trained_models", f"{name.replace(' ', '_').lower()}_model.joblib")
            image_path = os.path.join(settings.BASE_DIR,"static", "confusion_matrix", f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
            
            plot_simple_confusion_matrix(cm, classes=y.unique(), title=f'Confusion Matrix of {name}', image_path=image_path)
            plt.close()
            dump(model, model_path)
        except Exception as e:
            messages.error(request, f"Não foi possivel treinar {name}. Error: {e}")

    with open(os.path.join(settings.BASE_DIR, "train_algorithm", "model_accuracies.json"), "w") as f:
        json.dump(model_accuracies, f)
    messages.success(request, "modelos treinados com sucesso!")
    return redirect('algorithm')




