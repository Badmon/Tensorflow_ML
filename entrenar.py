import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convultion2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

data_entrenamiento='.data/entrenamiento'
data_validacion='.data/validacion'   

##parametros

epocas=20
altura, longitud = 100, 100
batch_size=32
pasos=1000
pasos_validacion=200
filtrosConv1=32
filtrosConv2=64
tamaño_filtro1=(3,3)
tamaño_filtro2=(2,2)
tamaño_pool=(2,2)
clases=3
lr=0.0005

##pre procesamiento de imagenes

entrenamiento_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)



