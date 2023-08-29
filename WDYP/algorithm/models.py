from django.db import models

class PlantingParameters(models.Model):
    p = models.FloatField()
    k = models.FloatField()
    temperature = models.FloatField()
    rainfall = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()

    def __str__(self):
        return f"Parameters: P={self.p}, K={self.k}, Temp={self.temperature}, Rainfall={self.rainfall}, Humidity={self.humidity}, pH={self.ph}"
