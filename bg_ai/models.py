import os
import uuid
from django.conf import settings
from django.db import models

# Create your models here.

def upload_name_generator(instance, filename):
    image = str(uuid.uuid4())[0:4]
    image_dir = os.path.join(settings.MEDIA_ROOT, settings.UPLOAD_DIR, image)
    os.makedirs(image_dir, exist_ok=True)
    path = settings.UPLOAD_DIR + '/' + image + '/' + filename
    return path


class Image(models.Model):
    uploaded_image = models.ImageField(upload_to='source')
    processed_image = models.ImageField(upload_to='processed')