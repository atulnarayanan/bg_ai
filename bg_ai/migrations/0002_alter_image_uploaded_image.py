# Generated by Django 4.0.1 on 2022-01-17 19:21

import bg_ai.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bg_ai', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='uploaded_image',
            field=models.ImageField(upload_to=bg_ai.models.upload_name_generator),
        ),
    ]
