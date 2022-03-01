


from gtsrb_inferrer import GtsrbInferrer
import numpy as np
from PIL import Image
from models.gtsrb_model import GtsrbModel

def gtsrb_model_deneme():
    print("baslangıc")
    deneme = GtsrbModel()
    print("data ekle")
    deneme.load_data()
    print("model olustur")
    deneme.build()
    print("egitim yap")
    deneme.train()
    print("bitti")

def gtsrb_inferrer_deneme():
    # inferrer deneme
    model = GtsrbInferrer()
    image = np.asarray(Image.open('//resources/test_sign/az_sayıda/00001.ppm')).astype(np.float32)
    image.resize((30, 30, 3))
    output = model.infer(image)
    print("reurn sonrası ", output['segmentation_output'])

    deger = output['segmentation_output']
    print(type(deger))
    deger = np.array(deger[0])
    most = deger.max()
    print(most)
    key = 0
    for x in range(len(deger)):
        if deger[x] == most:
            most = deger[x]
            key = x

    print(key)


gtsrb_model_deneme()
#gtsrb_inferrer_deneme()