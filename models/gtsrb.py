# -*- coding: utf-8 -*-
"""Abstract base model"""
"""
import numpy as np                               #basically an array
import pandas as pd                              #reading and analyze csv
import matplotlib.pyplot as plt                  #data visualisation
#import cv2                                       #comp. vision, image processing, uses numpy as images are 2D array(matrices)
import tensorflow as tf                          #creating neural network (collect, build, train, evaluate, predict)
from PIL import Image                            #manipulate images in python
import os                                       #directory control
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical           #one-hot encoding
import tqdm     
"""
from keras.models import Sequential #, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

import keras
from abc import ABC, abstractmethod
import os
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

path_main = "//"
#path_main = ""

class GtsrbModel(ABC):
#class GtsrbModel():
    """Abstract Model class that is inherited to all models"""

    #def __init__(self, cfg):
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    @abstractmethod
    def load_data(self):
        data_train = []
        labels = []
        classes = 43

        for i in range(classes):

            zeros = ""
            if i < 10:
                zeros = "0000"
            else:
                zeros = "000"
            path = path_main + "resources/Train/Images/" + zeros + str(i)
            print(path)
            images = os.listdir(path)
            print(len(images))
            for j in images:
                try:
                    image = Image.open(path + '/' + j)
                    image = image.resize((30, 30))
                    image = np.array(image)
                    data_train.append(image)
                    labels.append(i)
                except:
                    print("Error loading image ", j)

        self.X_train, self.X_test, y_train, y_test = train_test_split(data_train, labels, test_size=0.2, random_state=68)
        y_train = to_categorical(y_train, 43)
        y_test = to_categorical(y_test, 43)
        self.y_train=y_train
        self.y_test=y_test

    @abstractmethod
    def build(self):
        #model = keras.models.load_model("weights/gtsrb/Trafic_signs_model.h5")
        #model.load_weights("weights/gtsrb/Trafic_signs_model_weights.h5")
        input_sh = (30, 30, 3)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_sh))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model

    @abstractmethod
    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=2, validation_data=(self.X_test, self.y_test))



    def save(self):
        self.model.save(path_main+"weights/gtsrb/Trafic_signs_model.h5")
        self.model.save_weights(path_main+"weights/gtsrb/Trafic_signs_model_weights.h5")

    #@abstractmethod
    def evaluate(self):
        pass

