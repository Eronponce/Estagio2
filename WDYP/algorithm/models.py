from django.db import models

class PlantingInstance(models.Model):
    name = models.CharField(max_length=100)
    author = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class PlantingParameters(models.Model):
    instance = models.ForeignKey(PlantingInstance, on_delete=models.CASCADE)
    p = models.FloatField()
    k = models.FloatField()
    temperature = models.FloatField()
    rainfall = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()

    def __str__(self):
        return f"Parameters for {self.instance}: P={self.p}, K={self.k}, Temp={self.temperature}, Rainfall={self.rainfall}, Humidity={self.humidity}, pH={self.ph}"
