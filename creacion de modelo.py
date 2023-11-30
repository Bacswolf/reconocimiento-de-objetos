#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


# In[3]:


KMP_DUPLICATE_LIB_OK = True


# In[4]:


categorias = []
etiquetas = []
imagenes = []


# In[5]:


categorias = os.listdir("./train/train")
categorias.remove("datos.csv")
print(categorias)


# In[6]:


x = 0
for carpetas in categorias:
    for img in os.listdir("./train/train/" + carpetas):
        if img != "datos.csv":
            # print("./train/train/" + carpetas + "/" + img)
            image = Image.open("./train/train/" + carpetas + "/" + img).resize(
                (100, 100)
            )
            image = np.asarray(image)
            imagenes.append(image)
            etiquetas.append(x)
    x += 1


# In[7]:


imagenes = np.asanyarray(imagenes)


# In[8]:


imagenes.shape


# In[9]:


print(etiquetas)
etiquetas = np.asarray(etiquetas)


# In[2]:


plt.figure()
plt.imshow(imagenes[9])
plt.grid(False)
plt.show()


# In[10]:


modelo = tf.keras.Sequential(
    [
        # conv2d aplica filtros
        # relu funcion para alterar los valores de activacion del 0 al maximo
        tf.keras.layers.Conv2D(
            32, (3, 3), input_shape=(100, 100, 3), activation="relu"
        ),
        # se reduce la matriz con una matriz de 2x2 guardando solo el dato mayor procando que una matriz 6x6 pase a una 3x3
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # funcion para apagar el 50% de neuronas al entrenar
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=100, activation="relu"),
        # softmax se usa porque la respuesta puede quedar en una probabilidad
        tf.keras.layers.Dense(33, activation="softmax"),
    ]
)


# In[11]:


modelo.compile(
    # adam es el metodo del desenso del gradiente
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# In[12]:


import math

modelo.fit(imagenes, etiquetas, epochs=40, steps_per_epoch=math.ceil(120))


# In[13]:


modelo.save("tf_model_frutas2.h5")


# In[14]:


new_model = tf.keras.models.load_model("tf_model_frutas2.h5")


# In[35]:


im = 0
im = Image.open("./test/test/0213.jpg").resize((100, 100))
im = np.asarray(im)
# im=im[:,:,0]
im = np.array([im])
im.shape
test = im


# In[36]:


prediccion = new_model.predict(test)


# In[37]:


print(prediccion)


# In[38]:


categorias[np.argmax(prediccion[0])]


# In[ ]:
