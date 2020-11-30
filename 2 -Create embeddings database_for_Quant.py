#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:56:17 2020

@author: base

PLEASE ALSO CHECK THE JUPYTER FILE. IT ACTUALLY HAS MORE INFO
"""

from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
from numpy import asarray
from numpy import expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf
from tensorflow.keras import backend as K

import sys
import time

#model='tf220_all_int8'
#model='quantized_modelh5-15'
model='quantized_modelh5-13'
modelpath='/home/base/Documents/Git/Projekte/CelebFaceMatcher/compare the models delete/tflite/'

modelpath= modelpath + model + '.tflite'

Preprocessingversion=2 #1,2 oder 3  2 für tflite int8 ,3 für all_int8

"""
#-------------------------------------------------------------------------
# Load model'
"""
# Load TFLite model and allocate tensors.Beide modelle funktionieren
#Depending on the version of TF running, check where lite is set :
print(tf.__version__)
if tf.__version__.startswith ('1.'):
    print('lite in dir(tf.contrib)' + str('lite' in dir(tf.contrib)))

elif tf.__version__.startswith ('2.'):
    print('lite in dir(tf)? ' + str('lite' in dir(tf)))
   
print('workdir: ' + os.getcwd())
#os.chdir('/home/base/Documents/Git/Projekte/CelebFaceMatcher')

try: 
    interpreter = tf.lite.Interpreter(str(modelpath))   # input()    # To let the user see the error message
except ValueError as e:
    print("Error: Modelfile could not be found. Check if you are in the correct workdirectory. Errormessage:  " + str(e))
    import sys
    sys.exit()

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

PFAD = Path(Path.cwd() / 'All_croped_images/')
CelebFolders = next(os.walk(PFAD))[1]
EMBEDDINGS = pd.DataFrame()
ce=0

np.set_printoptions(threshold=sys.maxsize)# is needed to avoid ellipsis

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
            
    elif version == 3:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= np.round(91.4953).astype('uint8')
            x_temp[:, 1, :, :] -= np.round(103.8827).astype('uint8')
            x_temp[:, 2, :, :] -= np.round(131.0912).astype('uint8')
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= np.round(91.4953).astype('uint8')
            x_temp[..., 1] -= np.round(103.8827).astype('uint8')
            x_temp[..., 2] -= np.round(131.0912).astype('uint8')
    else:
        raise NotImplementedError

    return x_temp



for celeb in CelebFolders:#[0:2]:
    n = 0
    m = 0
    ce += 1
    print('-------------')
    print(str(celeb) + ' ' + str(ce) +' of '+str(len(CelebFolders))+ ' (' +str(ce*100/len(CelebFolders))+'%)')
    print('')
    filelist = next(os.walk(Path(PFAD / celeb)))[2]
    for f in filelist:
        n += 1
        m += 1
        img = cv2.imread(str(Path(PFAD / celeb / f)), cv2.IMREAD_COLOR)
        #cv2.imshow('ItsYou', img)
        # Make images the same as they were trained on in the VGGface2 Model
        # convert one face into samples
        if Preprocessingversion==1 or Preprocessingversion==2:
            samples = img.astype('float32')
            samples = expand_dims(samples, axis=0)
            samples = preprocess_input(samples, version=Preprocessingversion)
        elif Preprocessingversion==3:
            samples = expand_dims(img, axis=0)
            #prepare the face for the model, e.g. center pixels
            samples = preprocess_input(samples, version=Preprocessingversion).astype('int8')
        
        #input_data = convert(samples, 0, 255, np.uint8) #convert to unint8 for tflite quant
        interpreter.set_tensor(input_details[0]['index'], samples)
        interpreter.invoke()        
        features = np.ravel(interpreter.get_tensor(output_details[0]['index']))
        
        if EMBEDDINGS.empty:
            EMBEDDINGS = EMBEDDINGS.append({
                'Name': celeb, 
                'File': f, 
                'Embedding': features
                          },
                ignore_index=True,
                sort=False)                
            Only_embeddings =list([features])
            Only_name = list([celeb])
            Only_file = list([f])
        else:
            EMBEDDINGS = EMBEDDINGS.append(
                {
                    'Name': celeb,
                    'File': f,
                    'Embedding': features
                }, 
                ignore_index=True,
                sort=False)
            Only_embeddings.append(features)
            Only_name.append(celeb)
            Only_file.append(f)
        if n==1:
            print('finished ' + str(n) + ' of ' + str(len(filelist)))
        else:
            print('         ' + str(m) + ' of ' + str(len(filelist)))
            
filename_csv='EMBEDDINGS_' + model + '.csv'
filename_json='EMBEDDINGS_' + model + '.json'
EMBEDDINGS.to_csv(Path(Path.cwd() / filename_csv), index=False)
EMBEDDINGS.to_json(Path(Path.cwd() / filename_json))

#import csv

# def intocsv(filename, mylist):
#     with open(filename, 'w') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(mylist)

        
# intocsv('EMBEDDINGS_onlyEmbeddings_del', Only_embeddings)
# intocsv('EMBEDDINGS_onlyName_del', Only_name)
# intocsv('EMBEDDINGS_onlyFile_del', Only_file)

# # save numpy array as csv file
# from numpy import asarray
# from numpy import savetxt
# savetxt('AAAdata.csv', Only_embeddings, delimiter=',')

# print('I am done')