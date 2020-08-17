from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from disguise_face_app.views import index, login, register, home, logout

urlpatterns = [
    path('', index),
    path('login/', login),
    path('register/', register),
    path('home/', home),
    path('logout/', logout)
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
