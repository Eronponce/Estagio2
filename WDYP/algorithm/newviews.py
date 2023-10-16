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
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views import View
from .models import PlantingInstance, PlantingParameters, PredictionInfo

class ControlAlgorithm(View):
    @staticmethod
    @login_required
    def load_instance(request):
        instances = PlantingInstance.objects.all()
        instance_id = request.GET.get('instance_id')
        if instance_id:
            
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
            
            messages.success(request, 'Instancia e resultados carregados.')

            return render(request, "algorithm/algorithm.html", {
                'instances': instances,
                'instance': instance,
                'parameters': parameters,
                'predictions_info': predictions,
                'knn_variation': formatted_variation[0] if formatted_variation and formatted_variation[0] else None

            })
        return render(request, "algorithm/algorithm.html", {'instances': instances})
    

    @login_required
    def save_instance(self, request):
        return redirect('algorithm/algorithms.html')
    

    
    @login_required
    def delete_instance(request):
        instances = PlantingInstance.objects.all()
        instance_id = request.GET.get('instance_id')
        print(instance_id)
        instance = get_object_or_404(PlantingInstance, id=instance_id)
        instance.delete()
        messages.error(request, 'Instancia e resultados deletados.')
        return render(request, "algorithm/algorithm.html", {'instances': instances})

class ViewResults(View):
    @login_required
    def execute_algorithm(self, request):
       return redirect('algorithm')

    @login_required
    def show_results(self, request):
        return redirect('algorithm')

    @login_required
    def train(self, request):
        return redirect('algorithm')