#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:09:05 2020

@author: base
"""
import pandas as pd
from pathlib import Path
import os
import cv2
    
path=Path.cwd() / 'Free_com_Celeb_St/Free_com_Celeb'
face_cascade = cv2.CascadeClassifier('./Cascader/haarcascade_frontalface_alt.xml')
savefolder = Path.cwd() / 'Free_com_Celeb_St' / 'Free_com_Celeb_croped'
os.makedirs(savefolder, exist_ok=True)

p = 50        # Buffer for space around detected face to croping
width = 224
height = width
notfacelist=[]
Folderlist = next(os.walk(path))[1]
print(Folderlist)

for celeb in Folderlist:
    filelist = next(os.walk(Path(path / celeb)))[2]
    print(celeb)
    for f in filelist:  # Listing jpg files in this directory tree
        img = cv2.imread(str(Path(path / celeb / f)), cv2.IMREAD_COLOR)
        print(f)
        # Detect face
        faces_detected = face_cascade.detectMultiScale(img,
                                                       scaleFactor=1.1,
                                                       minNeighbors=4)
        if len(faces_detected) != 0:  # only if the cascader detected a face, otherwise error
            (x, y, w, h) = faces_detected[0]
            # create folderstructure with a new folder for each celebrity
            croppedpath = Path(savefolder / celeb)
            os.makedirs(croppedpath, exist_ok=True)
            filename = f'{croppedpath}/{f}'
            # Crop image to face
            img = img[y - p + 1:y + h + p, x - p + 1:x + w + p]  # use only the detected face; crop it
            if img.shape > (width, height) and img.size is not 0:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)  # resize the image to desired dimensions e.g., 256x256
                # Save croped image in folder
                cv2.imwrite(filename, img)  # save image in folder
#                 cv2.imshow(str(f), img)
            else:
                print('image to small or facebox out of image')
            
        else:
            print('no face detected. Apply Kill image')
            notfacelist.append(str(Path(path / celeb / f)))
            #cv2.imshow(str(f), img)
            #os.remove(str(Path(path / celeb / f)))
            
df = pd.DataFrame(notfacelist)
df.to_csv("./notfacelist.csv", sep=',',index=False)

#cv2.destroyAllWindows()
