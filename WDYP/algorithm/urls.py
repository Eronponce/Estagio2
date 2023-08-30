from django.urls import path
from . import views

urlpatterns = [
    path('', views.algorithm, name='algorithm'),
    path('save_parameters/', views.save_parameters, name='save_parameters'), 
    path('load_instance/', views.load_instance, name='load_instance'),
   
]
