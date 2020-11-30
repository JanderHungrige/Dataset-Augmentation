#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:58:51 2020

@author: base
"""

from keras_vggface_TF.vggfaceTF import VGGFace
from keras_vggface_TF.utils import preprocess_input
print('using tf.keras')
import tensorflow as tf
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot as plt
from scipy import stats

#from mtcnn import MTCNN
import time
from IPython import get_ipython 
get_ipython().run_line_magic('matplotlib', 'inline') 
#get_ipython().run_line_magic('matplotlib', 'qt') 
"""
#-------------------------------------------------------------------------
# Define Variables
"""
path=Path.cwd() / 'Free_com_Celeb_St'
faultpath=path / 'test_faulty'

model= Path('quantized_modelh5-15')
 

modelpath= path / (str(model) + '.tflite')
print(modelpath)

path=Path.cwd() / 'Free_com_Celeb_St/test_faulty'
CelebFolders = next(os.walk(path))[1]
print(CelebFolders)


EMBEDDINGS = pd.DataFrame()
ce=0

def faceembedding(YourFace,CelebDaten):
    Dist=[]
    for i in range(len(CelebDaten.File)):
        Celebs=np.array(CelebDaten.Embedding[i]) 
        Dist.append(np.linalg.norm(YourFace-Celebs))
    return Dist

def createEMBEDDINGS(EMBEDDINGS,celeb,f,featrues,MeanE=None,StdE=None,SemE=None):
        if EMBEDDINGS.empty:
            EMBEDDINGS = EMBEDDINGS.append({
                'Name': celeb, 
                'File': f, 
                'Embedding': features,
                'MeanEmb': MeanE,
                'StdEmb': StdE,
                'SemEmb': SemE
                          },
                ignore_index=True,
                sort=False)                

        else:
            EMBEDDINGS = EMBEDDINGS.append(
                {
                    'Name': celeb,
                    'File': f,
                    'Embedding': features,
                    'MeanEmb': MeanE,
                    'StdEmb': StdE,
                    'SemEmb': SemE
                }, 
                ignore_index=True,
                sort=False)

        return EMBEDDINGS


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

"""
#-------------------------------------------------------------------------
# Run analysis'
"""

for celeb in CelebFolders:#[0:2]:
    n = 0
    m = 0
    ce += 1
    print('-------------')
    print(str(celeb) + ' ' + str(ce) +' of '+str(len(CelebFolders))+ ' (' +str(ce*100/len(CelebFolders))+'%)')
    print('')
    EMBEDDINGS_temp=pd.DataFrame()
    filelist = next(os.walk(Path(path / celeb)))[2]
    time1=time()
    for f in filelist:
        n += 1
        m += 1
        img = cv2.imread(str(Path(path / celeb / f)), cv2.IMREAD_COLOR)
        #cv2.imshow('ItsYou', img)
        # Make images the same as they were trained on in the VGGface2 Model
        # convert one face into samples
        samples = img.astype('float32')
        samples = expand_dims(samples, axis=0)
        samples = preprocess_input(samples, version=2)

        
        #input_data = convert(samples, 0, 255, np.uint8) #convert to unint8 for tflite quant
        interpreter.set_tensor(input_details[0]['index'], samples)
        interpreter.invoke()        
        features = np.ravel(interpreter.get_tensor(output_details[0]['index']))
        
        EMBEDDINGS_temp=createEMBEDDINGS(EMBEDDINGS_temp,celeb,f,features)
        EMBEDDINGS=createEMBEDDINGS(EMBEDDINGS,celeb,f,features)

        if n==1:
            print('finished ' + str(n) + ' of ' + str(len(filelist)))
        else:
            print('         ' + str(m) + ' of ' + str(len(filelist)))
    print
    for idx in EMBEDDINGS_temp.index:
        #ME=faceembeddingNP(entry,EMBEDDINGS_temp)
        ME=faceembedding(EMBEDDINGS_temp.Embedding[idx],EMBEDDINGS_temp)

        EMBEDDINGS_temp.MeanEmb[idx]=np.mean(ME)
        EMBEDDINGS_temp.StdEmb[idx]=np.std(ME)
        EMBEDDINGS_temp.SemEmb[idx]=stats.sem(ME)
        
    print(EMBEDDINGS_temp.MeanEmb)
    OverallMean=np.mean(EMBEDDINGS_temp.MeanEmb)
    x=range(len(EMBEDDINGS_temp.Embedding))
    plt.errorbar(EMBEDDINGS_temp.File, 'MeanEmb', yerr='SemEmb', data=EMBEDDINGS_temp, fmt='o')
    plt.xticks(rotation=90)
    
    y=np.ones(len(EMBEDDINGS_temp.Embedding))*OverallMean
    plt.plot(EMBEDDINGS_temp.File,y)
    plt.title(celeb)
    plt.show()

#Removing the #files

#lowembeddings=[]
#removelist=[]
#for idx in EMBEDDINGS.index:
#    if EMBEDDINGS.MeanEmb[idx] > 99:
#        print('removing: ' + str(path / EMBEDDINGS.Name[idx] / EMBEDDINGS.File[idx]))
#        os.remove(str(path / EMBEDDINGS.Name[idx] / EMBEDDINGS.File[idx]))# Remove the file
#        EMBEDDINGS_modified=EMBEDDINGS.drop(EMBEDDINGS.index[idx]) # Remove that line from the Embeddings
#        lowembeddings.append(str(Path(path / celeb / f))) #Add info tofiel

#filename_csv='comEMBEDDINGS2_' + model + '.csv'
#filename_json='comEMBEDDINGS2_' + model + '.json'
#EMBEDDINGS_modified.to_csv(Path(Path.cwd() / filename_csv), index=False)
#EMBEDDINGS_modified.to_json(Path(Path.cwd() / filename_json))
#print(time.time()-time1)
