#Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import os
import numpy as np
from PIL import Image as Img
from PIL import ImageOps
import cv2
from tqdm import tqdm
from imutils import paths

from skimage.transform import resize
from skimage import io, transform

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from bg_ai.model_utils.data_loader import RescaleT
from bg_ai.model_utils.data_loader import ToTensor
from bg_ai.model_utils.data_loader import ToTensorLab
from bg_ai.model_utils.data_loader import SalObjDataset

from bg_ai.model import U2NET
from bg_ai.model import U2NETP
from django.conf import settings

def removebg(img_name, img_path):
    MODEL_NAME = 'u2netp'
    BASE_PATH = settings.BASE_DIR
    FILE_DICT = {
        "INPUT_DIRECTORY" : str(BASE_PATH) + '/files/source/',
        "TRANSPOSED_DIRECTORY" : str(BASE_PATH) + '/bg_ai/output/transposed/',
        "REMOVED_BACKGROUND_OUTPUT" : str(BASE_PATH) + '/bg_ai/output/bg_out/',
        "BLURRED_BACKGROUND_OUTPUT" : str(BASE_PATH) + '/bg_ai/output/blurred_out/',
        "MASKS" : str(BASE_PATH) + '/bg_ai/output/masks/',
        "MODEL_NAME" : 'u2netp',
        "MODEL_PATH" : str(BASE_PATH) + f'/bg_ai/model_weights/{MODEL_NAME}.pth',
    }
    
    #-------Creating Necessary Directories-----

    os.makedirs(FILE_DICT['TRANSPOSED_DIRECTORY'], exist_ok=True)
    os.makedirs(FILE_DICT["REMOVED_BACKGROUND_OUTPUT"], exist_ok=True)
    os.makedirs(FILE_DICT["BLURRED_BACKGROUND_OUTPUT"], exist_ok=True)
    os.makedirs(FILE_DICT['MASKS'], exist_ok=True)
    #-------Defining Utility Functions-----

    # normalize the predicted SOD probability map (SOD = Salient Object Detection)
    def normPRED(d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn
    
    def save_output(image_path, pred, desired_dir, image_name):

        # print(img_name)
        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Img.fromarray(predict_np*255).convert('RGB')
        # img_name = image_name.split(os.sep)[-1]
        image = io.imread(image_path)
        resized_image = im.resize((image.shape[1],image.shape[0]),resample=Img.BILINEAR)

        splitting = image_name.split(".")
        resized_image.save(desired_dir + splitting[0]+'.png')
    

    def convert_image(image_file):
        image = Img.open(image_file) # this could be a 4D array PNG (RGBA)
        original_width, original_height = image.size

        np_image = np.array(image)
        new_image = np.zeros((np_image.shape[0], np_image.shape[1], 3)) 
        # create 3D array

        for each_channel in range(3):
            new_image[:,:,each_channel] = np_image[:,:,each_channel]  
            # only copy first 3 channels.

        # flushing
        np_image = []
        return new_image
    
    #--------- 1. get model type and image path, transpose image with exif metadata and save in 'images' folder as a .jpg ---------

    model_name= FILE_DICT['MODEL_NAME'] # fixed as lightweight u2netp, change to 'u2net' for loading the 176 MB model


    #--------- 2. standardize format to '.jpg', from .JPG / .jpeg ---------
    
    imagename = img_name
    # inputdirectory = 'inp' # Default input image directory is set as "inp"
    # inp_dir = os.path.join(os.getcwd(), inputdirectory)
    image = Img.open(FILE_DICT["INPUT_DIRECTORY"] + imagename)
    # image = convert_image(FILE_DICT["INPUT_DIRECTORY"] + imagename)
    image = ImageOps.exif_transpose(image)
    inp_name = imagename[:-4]
    outp_name = str(inp_name) + '.jpg'
    image.save(FILE_DICT["TRANSPOSED_DIRECTORY"] + outp_name) 


    # image_dir = os.path.join(os.getcwd(), 'images/') # changed to 'images' directory which is populated while running the script
    # image_dir = os.path.join(os.getcwd(), FILE_DICT["TRANSPOSED_DIRECTORY"])
    # prediction_dir = os.path.join(os.getcwd(), FILE_DICT['MASKS']) # changed to 'masks' directory which is populated after the predictions
    # model_dir = os.path.join(os.getcwd(), FILE_DICT["MODEL_PATH"]) # path to u2netp / u2net pretrained weights

    img_name_list = [str(os.path.join(FILE_DICT['TRANSPOSED_DIRECTORY'], outp_name))]

    #---------- 3.dataloader ------------------------
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                        ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    
    # --------- 4. model define ---------
    if model_name == 'u2net': 
        # print("...Loading U2NET Model...")
        net = U2NET(3,1)
    elif model_name == 'u2netp':
        # print("...Loading U2NETP Model...")
        net = U2NETP(3,1)    
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(FILE_DICT['MODEL_PATH']))
        net.cuda()
    else:        
        net.load_state_dict(torch.load(FILE_DICT['MODEL_PATH'], map_location=torch.device('cpu')))

    net.eval()

    # --------- 5. Run Prediction To Get Mask and Save Mask ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        # save results to test_results folder
        if not os.path.exists(FILE_DICT['MASKS']):
            os.makedirs(FILE_DICT['MASKS'], exist_ok=True)
        save_output(image_path=img_path, pred=pred, desired_dir = FILE_DICT['MASKS'], image_name=img_name)

        del d1,d2,d3,d4,d5,d6,d7
    
    name = inp_name
    mask = load_img(FILE_DICT['MASKS']+name+'.png')

    # --------- 6. convert mask to numpy array and rescale(255 for RBG images) ---------
    RESCALE = 255
    out_img = img_to_array(mask)
    out_img /= RESCALE

    # --------- 7. Fuzzy Percentage Thresholding ---------

    fuzzy = ((0 < out_img) & (out_img <= 0.7)).sum()
    nonzero_count = np.count_nonzero(out_img)

    # define the cutoff threshold below which, background will be removed.
    THRESHOLD = 0.7

    # refine the output
    # out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    out_img = cv2.blur(out_img,(5,5))
    out_img = cv2.GaussianBlur(out_img, (7,7),0)

    # convert the rbg image to an rgba image and set the zero values to transparent
    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)

    # load and convert input to numpy array and rescale(255 for RBG images)
    input = load_img(FILE_DICT['TRANSPOSED_DIRECTORY'] + name +'.jpg')

    inp_img = img_to_array(input)
    inp_img /= RESCALE

    # since the output image is rgba, convert this also to rgba, but with no transparency
    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)

    # simply multiply the 2 rgba images to remove the backgound
    rem_back = (rgba_inp*rgba_out)
    rem_back_scaled = Img.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')
    return rem_back_scaled
    # rem_back_scaled.save(FILE_DICT["REMOVED_BACKGROUND_OUTPUT"]+name+".png")

    # # BLURREDBACKGROUND
    # blurredFullImg = cv2.GaussianBlur(rgba_inp, (101,101), 0)
    # foreground = (rgba_inp*rgba_out)
    # background = ((1.0 - rgba_out)*blurredFullImg)
    # blur_bg_out = foreground + background
    # blur_bg_scaled = Img.fromarray((blur_bg_out*RESCALE).astype('uint8'), 'RGBA')
    # print(type(blur_bg_scaled))
    # blur_bg_scaled.save(FILE_DICT["BLURRED_BACKGROUND_OUTPUT"]+name+".png")
