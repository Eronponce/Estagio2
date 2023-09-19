from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
   
    path('', include('algorithm.urls')),
    path('', include('authentication.urls')),
    # ... other project URLs ...
]
