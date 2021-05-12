from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import pickle
import time
import cv2
import os
import dlib
import shutil

model_path = 'liveness_micro_92.h5'

# load face detector
hog_detector = dlib.get_frontal_face_detector()

print("[INFO] loading trained model liveness detector...")
model = load_model(model_path)
currentFolder = 'C:/Users/smoreyra/Desktop/gusa_ejemplo'
#currentFolder = 'C:/Users/smoreyra/Desktop/ANTISPOOFING/imiroinica-celeba/liveness-master/4_liveness net/modelo_92_val_ac/validacion/falsos_negativos'
imagePaths = list(paths.list_images(currentFolder))

fake = 0
real = 0
noFace = 0
multipleFaces = 0
total = 0
print("Testing Photos")
for imagePath in imagePaths:
    try:
        image = dlib.load_rgb_image(imagePath)
    except:
        continue
    (H, W, D) = image.shape
    try:
        dets = hog_detector(image, 1)
    except:
        continue
    if(len(dets) == 0):
        print('None face detected')
        total = total + 1
        noFace = noFace + 1
        #try:
        #   shutil.move(imagePath, 'C:/Users/smoreyra/Desktop/ejemplo-gusa/no_face')
        #except:
        #   continue
        #continue
    if(len(dets) > 1):
        print('Multiple faces detected')
        multipleFaces = multipleFaces + 1
        #try:
        #   shutil.move(imagePath, 'C:/Users/smoreyra/Desktop/ejemplo-gusa/multiple_face')
        #except:
        #   continue
        #continue
    try:
        d = dets[0]
    except:
        continue
    w = d.right() - d.left()
    h = d.bottom() - d.top()
    cx = int((d.right() + d.left()) / 2)
    cy = int((d.bottom() + d.top()) / 2)
    w = w * 2
    h = h * 2
    top = cy - int(h//2)
    bottom = cy + int(h//2)
    right = cx + int(w//2)
    left = cx - int(w//2)
    top = max(top, 0)
    bottom = min(bottom, H-1)
    left = max(left, 0)
    right = min(right, W-1)
    #face_crop = image[d.top():d.bottom(), d.left():d.right()]
    face_crop = image[top:bottom, left:right]
    if(face_crop.size == 0):
        print("NO CROP")
        try:
           shutil.move(imagePath, 'C:/Users/smoreyra/Desktop/ejemplo-gusa/real')
        except:
           continue
        continue
    total = total + 1
    face_crop = cv2.resize(face_crop, (224, 224))
    #newPath='prueba/crop' + str(total) + '.jpg'
    #cv2.imwrite(newPath, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
    face = img_to_array(face_crop)
    face = np.expand_dims(face, axis=0)
    preds = model.predict(face)
    real_pred = preds[0]
    print(preds)
    predicted_class = (real_pred >= 0.99958) * 1.0
    if(predicted_class == 1):
        print(imagePath, ' :real ', "PRED: ", real_pred, " ")
        real = real + 1
        #try:
        #   shutil.move(imagePath, 'C:/Users/smoreyra/Desktop/ejemplo-gusa/real')
        #except:
        #   continue
    else:
        print(imagePath, ' :fake ', "PRED: ", real_pred, " ")
        fake = fake + 1
        #try:
        ##    shutil.move(
        ##    imagePath, 'C:/Users/smoreyra/Desktop/ejemplo-gusa/fake')
        ##except:
         #   continue

    # print(imagePath)
print("Total Real: ", real)
print("Total Fake: ", fake)
print("No Face", noFace)
print("Multiple Faces", multipleFaces)
print("Total", total)
