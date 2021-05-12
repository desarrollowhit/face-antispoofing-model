# import the necessary packages
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
import dlib
import numpy as np
import argparse
import cv2
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model_path = 'liveness_micro_92.h5'

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = (1e-3)/4
BS = 48
EPOCHS = 1000

# construct the training image generator for data augmentation
train_datagen=ImageDataGenerator()

train_generator=train_datagen.flow_from_directory('../../db_faces/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='binary',
                                                 shuffle=True)

validation_generator=train_datagen.flow_from_directory('../../db_faces/test',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='binary',
                                                 shuffle=False)
 
labels = (train_generator.class_indices)
print(labels)
# initialize the optimizer and model
adam_opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS)

try:
    model = load_model(model_path)
except:
    model = LivenessNet.build(width=224, height=224, depth=3,
        classes=len(labels))

model.summary()

print("[INFO] compiling model...")
#configure the learning process
model.compile(loss="binary_crossentropy", optimizer= adam_opt,
	metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 50)

step_size_train = train_generator.n//train_generator.batch_size
step_size_validation = validation_generator.samples // validation_generator.batch_size

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))

for i in range(EPOCHS):
    print("[INFO] Class indices")
    labels = (train_generator.class_indices)
    print(labels)

    Y_pred = 1.0 * (model.predict(validation_generator, validation_generator.samples // BS + 1) >= 0.995)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, Y_pred))
    print('Classification Report')
    target_names = ['Fake', 'Real']
    print(classification_report(validation_generator.classes, Y_pred))

    #H = model.fit(train_generator,
    #                steps_per_epoch=step_size_train,
    #                validation_data = validation_generator,
    #                validation_steps = step_size_validation,
    #                epochs=1,
    #                callbacks = [early_stopping]
    #                    )

    # save the network to disk
    #print("[INFO] serializing network to '{}'...".format(model_path))
    #model.save(model_path)
