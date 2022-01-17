from wsgiref import validate
from rest_framework import serializers
from bg_ai.models import Image

class ImageSerializer(serializers.ModelSerializer):


    class Meta:
        model = Image
        fields = ['uploaded_image','processed_image']
        read_only_fields = ['processed_image']
    
    