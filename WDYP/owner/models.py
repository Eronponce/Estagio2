from django.db import models
from django.forms import ModelForm, DateField, DateInput

class Owner(models.Model):
    name = models.CharField(max_length=100)
    cpf = models.IntegerField()
    birth = models.DateField()

    def __str__(self):
        return self.name

class Property(models.Model):
    name = models.CharField(max_length=100)
    cep = models.CharField(max_length=10)
    city = models.CharField(max_length=100)   
    state = models.CharField(max_length=100)
    property_number = models.IntegerField()
    square_meters = models.IntegerField()
    ownerid = models.ForeignKey(Owner, on_delete=models.CASCADE)
    def __str__(self):
        return self.name



class OwnerForm(ModelForm):
    class Meta:
        model = Owner
        fields = ['name', 'cpf', 'birth']
    birth = DateField(widget=DateInput(attrs={'type': 'date'}))
class PropertyForm(ModelForm):
    class Meta:
        model = Property
        fields = ['name', 'cep', 'city', 'state', 'property_number', 'square_meters', 'ownerid']
