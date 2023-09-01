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
