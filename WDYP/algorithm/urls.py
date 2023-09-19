from django.urls import path
from . import views

urlpatterns = [
    path('algorithm', views.algorithm, name='algorithm'),
    path('save_parameters/', views.save_parameters, name='save_parameters'), 
    path('algorithm/', views.train, name='train'),
    path('load_instance/', views.load_instance, name='load_instance'),
]
