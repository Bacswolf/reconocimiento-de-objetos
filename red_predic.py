import tensorflow as tf
import numpy as np
from PIL import Image
import os

class Red_pred:
    def __init__(self,ruta):
        self.ruta=ruta
        
    def predecir(self):
        categorias = os.listdir('train\\train\\')
        categorias.remove("datos.csv")
        new_model = tf.keras.models.load_model("tf_model_frutas2.h5")
        img=0
        img=Image.open(self.ruta).resize((100,100))
        img=np.asarray(img)
        #img=img[:,:,0]
        img=np.array([img])
        img.shape
        prediccion=new_model.predict(img)
        return categorias[np.argmax(prediccion[0])]
        
        
        