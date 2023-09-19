from django.urls import path
from .views import UserLoginView, UserLogoutView 
from .views import register
from . import views
urlpatterns = [
    path('login/', UserLoginView.as_view(), name='login'),
    path('logout/', UserLogoutView.as_view(), name='logout'),
    path('register/', register, name='register'),
     path('', views.home_view, name='home'),
]
