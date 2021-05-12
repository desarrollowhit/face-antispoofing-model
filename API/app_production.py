from flask import Flask, jsonify, request
from keras.preprocessing.image import img_to_array
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
from pandas import DataFrame
import base64
import io
from imageio import imread
import math
from gevent.pywsgi import WSGIServer

def create_app(testing: bool = True):
    i = 0
    app = Flask(__name__)

    #CONTADOR DE PRUEBA

    api_key = "1a0bb17e-cfe9-4283-b59a-cd97db897231"
    print("[INFO] loading trained model liveness detector...")
    model_path = 'liveness_micro_92.h5'
    model = load_model(model_path)

    # load face detector
    hog_detector = dlib.get_frontal_face_detector()

    def testApiKey(apiKey):
        if(api_key == apiKey):
            return True
        return False


    def truncate(number, digits):
        stepper = 10.0 ** digits
        if (number < 0.0001):
            return '{:.6f}'.format(number)
        else:
            return math.trunc(stepper * number) / stepper


    @app.route('/test-liveness', methods=['POST'])
    def testLiveness():
        global i 
    

        if(testApiKey(request.headers.get('x-api-key'))):

            try:
                decoded = base64.b64decode(request.json["img"])
                img = imread(io.BytesIO(decoded))
            except:
                #Unsupported image type, must be 8bit gray or RGB image.
                return jsonify({"status": "error", "resultado": "decodification_error"})

            try:
                dets = hog_detector(img, 1)
            except:
                return jsonify({"status": "error", "resultado": "no_supported"})
            if(len(dets) == 0):
                print('None face detected')
                return jsonify({"status": "error", "resultado": "no_face"})
            if(len(dets) > 1):
                print('Multiple faces detected')
                return jsonify({"status": "error", "resultado": "multiple_face"})
            else:
                d = dets[0]
                (H, W, D) = img.shape
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
                face_crop = img[top:bottom, left:right]
                if(face_crop.size == 0):
                    print("NO CROP")
                    return jsonify({"status": "error", "resultado": "crop_error"})
                face_crop = cv2.resize(face_crop, (224, 224))
                face = img_to_array(face_crop)
                face = np.expand_dims(face, axis=0)
                preds = model.predict(face)
                predicted_class_indices = np.argmax(preds, axis=1)
                response = {
                    # '{:.6f}'.format(float(preds[0])),
                    "liveness": truncate(float(preds[0]), 5),
                    "resultado": "",
                    "status": "ok"
                }
                predicted_class = (preds[0] >= 0.99958) * 1.0
                i = i + 1   
                print(i)
                if(predicted_class == 1):
                    print('real ', "PREDS: ", preds, " ")
                    response["resultado"] = "real"
                else:
                    print('fake ', "PREDS: ", preds, " ")
                    response["resultado"] = "fake"
            return jsonify(response)
        else:
            return jsonify({"status": "error", "message": "unauthorized"})

    return app