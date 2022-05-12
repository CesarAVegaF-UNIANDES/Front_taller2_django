from django.urls import path

from Front_django.view import recomendacionesView, loginView

urlpatterns = [
    path('recomendaciones/', recomendacionesView),
    path('', loginView),
]
