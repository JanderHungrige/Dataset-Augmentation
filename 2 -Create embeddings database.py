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
import sys
import time

resnet50_features = VGGFace(model='resnet50',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='avg')  # pooling: None, avg or max

PFAD = Path(Path.cwd() / 'All_croped_images/')
CelebFolders = next(os.walk(PFAD))[1]
EMBEDDINGS = pd.DataFrame()
ce=0

np.set_printoptions(threshold=sys.maxsize)# is needed to avoid ellipsis

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

        # Make images the same as they were trained on in the VGGface2 Model
        # convert one face into samples
        pixels = img.astype('float32')
        samples = expand_dims(pixels, axis=0)
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)

        features = np.ravel(resnet50_features.predict(samples))
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

EMBEDDINGS.to_csv(Path(Path.cwd() / 'EMBEDDINGS_del.csv'), index=False)
EMBEDDINGS.to_json(Path(Path.cwd() / 'EMBEDDINGS_del.json'))

import csv

def intocsv(filename, mylist):
    with open(filename, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(mylist)

        
intocsv('EMBEDDINGS_onlyEmbeddings_del', Only_embeddings)
intocsv('EMBEDDINGS_onlyName_del', Only_name)
intocsv('EMBEDDINGS_onlyFile_del', Only_file)

# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
savetxt('AAAdata.csv', Only_embeddings, delimiter=',')

print('I am done')