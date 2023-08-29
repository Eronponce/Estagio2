from django.urls import path
from . import views

urlpatterns = [
    path('', views.algorithm, name='algorithm'),
    path('save_parameters/', views.save_parameters, name='save_parameters'),  # Certifique-se de que esta linha está presente
    # ...
]
