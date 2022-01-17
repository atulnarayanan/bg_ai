from django.urls import path, include
from bg_ai import views

urlpatterns = [
    path('upload/',views.UploadAPI.as_view()),
    path('<int:pk>/retrieve/',views.ProcessAPI.as_view()),
]