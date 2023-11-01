from django.urls import path
from . import views
from owner import views as owner_views
urlpatterns = [
    path('algorithm', views.algorithm, name='algorithm'),
    path('save_parameters/', views.save_parameters, name='save_parameters'), 
    path('algorithm/', views.train, name='train'),
    path('load_instance/', views.load_instance, name='load_instance'),
    path('owner/', owner_views.controlOwner, name='owner'),
    path('property/', owner_views.controlProperty, name='property'),
]
