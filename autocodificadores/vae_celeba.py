#from tensorflow.keras import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import numpy as np
import autocodificador_utils as utils
import tensorflow as tf
# Cargar el conjunto de datos CelebA
#path = "C:\Users\Alejandro\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\celeba-dataset\img_align_celeba\img_align_celeba"

#train_data = image_dataset_from_directory(
#    path, labels=None,
#    color_mode = "rgb", image_size=(64, 64),
#    batch_size=128, shuffle=True, seed=42, interpolation="bilinear"
#    ) # Cargar el conjunto de datos CelebA
def preprocess(imgs):
    imgs = imgs / 255.0 # Normalizar las imágenes entre 0 y 1
    return imgs # Devolver las imágenes preprocesadas
#train =train_data.map(lambda x: preprocess(x)) # Preprocesar las imágenes de entrenamiento

#encoder
encoder_input = layers.Input(shape=(32, 32, 3), name='encoder_input')
x = layers.Conv2D(128,(3,3),strides=2,padding="same")(encoder_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128,(3,3),strides=2,padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128,(3,3),strides=2,padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128,(3,3),strides=2,padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
sbf = K.int_shape(x)[1:] # Obtener la forma de salida del codificador
x = layers.Flatten()(x) # Aplanar la salida del codificador
mean = layers.Dense(200, name='mean')(x) # Capa densa para la media
log_var = layers.Dense(200, name='log_var')(x) # Capa densa para la varianza logarítmica
z = utils.Sampling()([mean, log_var]) # Muestreo de la distribución normal
encoder = models.Model(encoder_input, [mean, log_var, z], name='encoder') # Definir el modelo del codificador
encoder.summary() # Resumen del modelo
