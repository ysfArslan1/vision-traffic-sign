
import requests
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

ENDPOINT_URL = "http://127.0.0.1:5000/infer"

def infer():
    image = np.asarray(Image.open('resources/test_sign/az_sayıda/00000.ppm')).astype(np.float32)
    image.resize((30, 30, 3))
    data ={'image': image.tolist()}
    response = requests.post(ENDPOINT_URL, json = data)
    print(response.raise_for_status())
    print(response.json())

if __name__ =="__main__":
    infer()

"""
import requests
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

ENDPOINT_URL = "http://127.0.0.1:5000/infer"

def infer():
    image = np.asarray(Image.open('resources/yorkshire_terrier.jpg')).astype(np.float32)
    data ={'image': image.tolist()}
    response = requests.post(ENDPOINT_URL, json = data)
    print(response.raise_for_status())
    print(response.json())


if __name__ =="__main__":
    infer()

"""