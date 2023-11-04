from django.urls import path
from . import views

urlpatterns  = [
path('edit/<int:id>', views.update_owner, name='edit_owner'),
path('delete/<int:id>', views.delete_owner, name='delete_owner'),
path('owner', views.controlOwner, name='owner_page'),
]