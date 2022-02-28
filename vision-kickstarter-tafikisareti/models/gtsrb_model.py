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
    #@abstractmethod
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
            path = "/home/arslan/Desktop/vision-kickstarter-tafikisareti/resources/Train/Images/" + zeros + str(i)
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

    #@abstractmethod
    def build(self):
        model = keras.models.load_model("weights/gtsrb/Trafic_signs_model.h5")
        model.load_weights("weights/gtsrb/Trafic_signs_model_weights.h5")
        self.model = model

    #@abstractmethod
    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=2, validation_data=(self.X_test, self.y_test))

    def save(self):
        self.model.save("weights/gtsrb/Trafic_signs_model.h5")
        self.model.save_weights("weights/gtsrb/Trafic_signs_model_weights.h5")

    #@abstractmethod
    def evaluate(self):
        pass

