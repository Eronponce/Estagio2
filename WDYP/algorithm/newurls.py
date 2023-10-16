# algorithm/urls.py
from django.urls import path
from .views import ControlAlgorithm, ViewResults

urlpatterns = [
    path('load_instance/' , ControlAlgorithm.load_instance, name='load_instance'),
    path('save_instance/', ControlAlgorithm.save_instance, name='save_instance'),
    path('delete_instance/', ControlAlgorithm.delete_instance, name='delete_instance'),
    path('execute_algorithm/', ViewResults.execute_algorithm, name='execute_algorithm'),
    path('show_results/', ViewResults.show_results, name='show_results'),
    path('train/', ViewResults.train, name='train'),
]