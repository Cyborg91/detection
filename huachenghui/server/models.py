from django.db import models

# Create your models here.
class Profile2(models.Model):
    picture_test = models.ImageField(upload_to='test_pictures')

    class Meta:
        db_table = "profile"


def __str__(self):
    return self.name
