import tensorflow as tf
import numpy as np
import sys
import os
import keras
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from utils.config import Config
from configs.config import CFG

class GtsrbInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.image_size = (30,30,3)

        self.saved_path = 'weights/gtsrb/Trafic_signs_model.h5'
        self.model = keras.models.load_model(self.saved_path)

    def infer(self, image=np.array):
        print("infer bas ",type(image))
        tensor_image = image

        tensor_image = np.array(tensor_image)
        print("reshape oncesi shape ",tensor_image.shape)
        print(self.model.summary())
        L = []
        L.append(tensor_image)
        tensor_image = np.array(L)
        print(tensor_image.shape)
        self.model.load_weights("weights/gtsrb/Trafic_signs_model_weights.h5")

        pred = self.model.predict(tensor_image)

        pred = pred.tolist()

        return {'segmentation_output':pred}
