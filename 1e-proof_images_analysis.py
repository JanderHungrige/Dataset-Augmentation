#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:30:39 2020

@author: base
"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:58:51 2020

@author: base
"""

from keras_vggface_TF.vggfaceTF import VGGFace
from keras_vggface_TF.utils import preprocess_input
print('using tf.keras')
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from numpy import asarray
from numpy import expand_dims
from scipy import stats
import time

#get_ipython().run_line_magic('matplotlib', 'qt')
"""
#-------------------------------------------------------------------------
# Define Variables
"""
basepath=Path.cwd() / 'Free_com_Celeb_St'
faultpath=basepath / 'test_faulty'

model= 'quantized_modelh5-15'

modelpath= basepath / (model + '.tflite')
print(modelpath)

path=basepath / 'Free_com_Celeb_croped'
print(path)
#path=Path.cwd() / 'Free_com_Celeb_St/test_faulty'

CelebFolders = next(os.walk(path))[1]

print(CelebFolders)

EMBEDS=pd.read_csv('/home/base/Documents/Git/Projekte/CelebFaceMatcher/Embeddings/EMBEDDINGS_tf220_all_int8.csv')

EMBEDDINGS = pd.DataFrame()
ce=0

def CreateEudist(One_image,CelebDaten, name):
    Dist=[]
    Auszug=CelebDaten.loc[CelebDaten.Name==name ,['Embedding'] ] # extract the embeddings only for that celeb name
    for i in range(len(Auszug)):
        Dist.append(np.linalg.norm(One_image-CelebDaten.Embedding[i]))
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
# Load Embeddings from non licensed images'
"""
Embeddings_golden_truth=pd.read_json(basepath / 'EMBEDDINGS_tf220_all_int8.json')


"""
#-------------------------------------------------------------------------
# Load model'
"""
# Load TFLite model and allocate tensors.
try: 
    interpreter = tflite.Interpreter(str(modelpath))   # input()    # To let the user see the error message
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
time1=time.time()
for celeb in CelebFolders:
    n = 0
    m = 0
    ce += 1
    print('-------------')
    print(str(celeb) + ' ' + str(ce) +' of '+str(len(CelebFolders))+ ' (' +str(ce*100/len(CelebFolders))+'%)')
    print('')
    EMBEDDINGS_temp=pd.DataFrame()
    filelist = next(os.walk(Path(path / celeb)))[2]
    time2=time.time()
    for f in filelist[0:2]:
        n += 1
        m += 1
        img = cv2.imread(str(Path(path / celeb / f)), cv2.IMREAD_COLOR)
        # Make images the same as they were trained on in the VGGface2 Model
        samples = img.astype('float32')
        samples = expand_dims(samples, axis=0)
        samples = preprocess_input(samples, version=2)
        #Create EMbeddings with tflite model
        interpreter.set_tensor(input_details[0]['index'], samples)
        interpreter.invoke()
        features = np.ravel(interpreter.get_tensor(output_details[0]['index']))
        EMBEDDINGS_temp=createEMBEDDINGS(EMBEDDINGS_temp,celeb,f,features)
        EMBEDDINGS     =createEMBEDDINGS(EMBEDDINGS,celeb,f,features)

        if n==1:
            print('finished ' + str(n) + ' of ' + str(len(filelist)))
        else:
            print('         ' + str(m) + ' of ' + str(len(filelist)))
            
    print(time.time()-time2)
    # Now compare for that folder the Eudist against the ground truth Eudist, or inbetween the images in the folder
    for idx in EMBEDDINGS_temp.index:
        #ME=CreateEudist(EMBEDDINGS_temp.Embedding[idx],EMBEDDINGS_temp,name=f) # Compare images within itselve 
        ME=CreateEudist(EMBEDDINGS_temp.Embedding[idx],Embeddings_golden_truth,name=celeb) # Compare images with golden truth 

        EMBEDDINGS_temp.MeanEmb[idx]=np.mean(ME)
        EMBEDDINGS_temp.StdEmb[idx]=np.std(ME)
        EMBEDDINGS_temp.SemEmb[idx]=stats.sem(ME)
        
    print(time.time()-time1)
    #saving the Embeddings after each folder just as backup
    EMBEDDINGS.to_json(Path(path / ('embeddings/comEMBEDDINGS_' + celeb + '.json')))
    EMBEDDINGS_temp.to_json(path / ('embeddings/comEMBEDDINGS_only_folder' + celeb + '.json'))

# Before we saved the embeddings in each folder, now we get the total embeddings as well and calculate the means
time3=time.time()
for idx in EMBEDDINGS.index:
    ME=CreateEudist(EMBEDDINGS.Embedding[idx],Embeddings_golden_truth,name=celeb) # Compare images with golden truth 
    EMBEDDINGS.MeanEmb[idx]=np.mean(ME)
    EMBEDDINGS.StdEmb[idx]=np.std(ME)
    EMBEDDINGS.SemEmb[idx]=stats.sem(ME)
    
EMBEDDINGS.to_json(Path(path/ ('embeddings/comEMBEDDINGS_all.json')))
print(time.time()-time3)

# Removing the files

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
