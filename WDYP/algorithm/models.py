from django.db import models


class PlantingInstance(models.Model):
    name = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    def __str__(self):
        return self.name


class PlantingParameters(models.Model):
    instance = models.ForeignKey(PlantingInstance, on_delete=models.CASCADE)
    n = models.FloatField(default=0)
    p = models.FloatField(default=0)
    k = models.FloatField(default=0)
    temperature = models.FloatField(default=0)
    humidity = models.FloatField(default=0)
    ph = models.FloatField(default=0)
    rainfall = models.FloatField(default=0)
    def __str__(self):
        return f"Parameters for {self.instance}: P={self.p}, K={self.k}, Temp={self.temperature}, Rainfall={self.rainfall}, Humidity={self.humidity}, pH={self.ph}"
from django.db import models

class PredictionInfo(models.Model):
    instance = models.ForeignKey(PlantingInstance, on_delete=models.CASCADE)
    model_number = models.IntegerField()
    prediction = models.CharField(max_length=100)
    max_confidence = models.FloatField()
    algorithm_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    def __str__(self):
        return f"Prediction by {self.algorithm_name} for {self.instance}: {self.prediction} (Confidence: {self.max_confidence}, Accuracy: {self.accuracy})"


class FormattedVariationKNN(models.Model):
    instance = models.ForeignKey(PlantingInstance, on_delete=models.CASCADE)
    variation = models.CharField(max_length=100)  # Using comma-separated values
    def __str__(self):
        return f"Formatted Variation for {self.instance}: {self.variation}"
