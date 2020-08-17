from django.db import models


class User(models.Model):

    class Meta:
        db_table = 'user'

    full_name = models.CharField(max_length=50, null=True)
    contact = models.CharField(max_length=15, null=True)
    email = models.CharField(max_length=75, null=True)
    password = models.CharField(max_length=100, null=True)
