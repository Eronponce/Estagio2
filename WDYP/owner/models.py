from django.db import models


class Owner(models.Model):
    name = models.CharField(max_length=100)
    cep = models.IntegerField()
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    property_number = models.IntegerField()
    square_meters = models.IntegerField()
    def __str__(self):
        return self.name

class Property(models.Model):
    owner = models.ForeignKey(Owner, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    cpf = models.IntegerField()
    birth_date = models.DateField()

    def __str__(self):
        return self.name
