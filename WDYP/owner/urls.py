from django.urls import path
from . import views

urlpatterns  = [
path('owner/', views.controlOwner, name='owner'),
path('property/', views.controlProperty, name='property')
]