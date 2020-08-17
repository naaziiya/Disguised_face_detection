from django.shortcuts import render, redirect
import shutil
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Q
from disguise_face_app.models import User
from disguise_face_app import predict
import pandas as pd


def index(request):
    return render(request, 'index.html')


def logout(request):
    return redirect('/')


def login(request):
    if request.method == 'POST':
        login_id = request.POST['loginId']
        password = request.POST['password']

        user = User.objects.filter(email=login_id, password=password).first()

        if user is None:
            return render(request, 'login.html', {'error': 'Invalid Login Credentials'})
        else:
            return render(request, 'home.html')
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':

        name = request.POST['name']
        contact = request.POST['contact']
        email = request.POST['email']
        password = request.POST['password']

        user = User.objects.filter(
            Q(contact=contact) | Q(email=email)
        ).first()

        if user != None and user.contact == contact:
            return render(request, 'register.html', {'error': 'Duplicate Contact'})

        if user != None and user.email == email:
            return render(request, 'register.html', {'error': 'Duplicate Email'})

        user = User(full_name=name, contact=contact,
                    email=email, password=password)
        user.save()

        return render(request, 'register.html', {'mes': 'Registered Successfully'})
    else:
        return render(request, 'register.html')


def home(request):

    if request.method == "GET":
        return render(request, 'home.html')

    if request.method == "POST":
        image = request.FILES['image']

        shutil.rmtree(os.getcwd() + '\\media')

        path = default_storage.save(
            os.getcwd() + '\\media\\input.jpg', ContentFile(image.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)

        result = predict.process()

        print(result)

    return render(request, "result.html", {'result': result})
