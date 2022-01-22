from django.shortcuts import render
from rest_framework import generics, viewsets
from rest_framework.parsers import MultiPartParser, JSONParser, FormParser

from bg_ai import serializer
from bg_ai.models import Image
from bg_ai.function import removebg

class UploadAPI(generics.ListCreateAPIView):
    parser_classes = [MultiPartParser, JSONParser, FormParser]
    serializer_class = serializer.ImageSerializer
    queryset = Image.objects.all()

    def post(self, request, *args, **kwargs):
        print('entry test')
        return super().post(request, *args, **kwargs)

class ProcessAPI(generics.RetrieveAPIView):
    parser_classes = [MultiPartParser, JSONParser, FormParser]
    serializer_class = serializer.ImageSerializer
    queryset = Image.objects.all()

    def retrieve(self, request, *args, **kwargs):
        print('test')
        image_id = self.kwargs.get('pk')
        print(image_id)
        image = Image.objects.get(id=image_id)
        print(image.uploaded_image.path)
        print(image.uploaded_image.name.split('source/')[1])
        processed_image = removebg(image.uploaded_image.name.split('source/')[1], img_path=image.uploaded_image.path)
        # image.processed_image.save()
        return super().retrieve(request, *args, **kwargs)