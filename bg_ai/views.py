from PIL import Image
import io
import gc
from django.core.files.base import ContentFile
from django.core.files import File
from django.core.files.uploadedfile import InMemoryUploadedFile

from django.conf import settings
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
        print(image.uploaded_image.name.split('source/')[1].split)
        ext = image.uploaded_image.name.split('source/')[1].split('.')[0]
        print(type(ext))
        filename = image.uploaded_image.name.split('source/')[1].split('.')[0]
        print(filename)
        final_filename = filename + '_processed.png'
        print(final_filename)
        processed_image = removebg(image.uploaded_image.name.split('source/')[1], img_path=image.uploaded_image.path)
        print(processed_image)


        img_io = io.BytesIO()
        processed_image.save(img_io, 'png')
        # print(processed_image)
        # temp_name = image.uploaded_image.name.split('source/')[1]+'_'+'processed'+'.png'
        # image.processed_image.delete(save=False)  

        # image.processed_image.save(
        #     temp_name,
        #     content=ContentFile(new_image_io.getvalue()),
        #     save=False
        # )

        # image.processed_image.save(img2_io, InMemoryUploadedFile(
        #     processed_image,       # file
        #     None,               # field_name
        #     image.uploaded_image.name,           # file name
        #     'image/png',       # content_type
        #     img2_io.tell,  # size
        #     None)               # content_type_extra
        # )

        image.processed_image.save(final_filename,File(img_io),save=True)
        gc.collect()
        # image.processed_image.save()
        return super().retrieve(request, *args, **kwargs)


        # # here is how i didi it in case of any one needed it:
        # img = Image.open('test.png')  #first we open an image with PIL or maybe you have it from uploaded file already(you should import PIL and Image)
        # img_io = io.BytesIO()            #creat in memory object by io (you should import io)
        # img.save(img_io,format='png') #save your PIL image object to memory object you created by io
        # #you should import InMemoryUploadedFile
        # thumb = InMemoryUploadedFile(img2_io, None, 'foo2.jpeg', 'image/jpeg',thumb_io.seek(0,os.SEEK_END), None) #give your file to InMemoryUploadedFile to create django imagefield object
        # #take look at this link to find out to import what things: