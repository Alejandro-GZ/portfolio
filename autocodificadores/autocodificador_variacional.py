import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import numpy as np
import autocodificador_utils as utils
    
encoder_input = layers.Input(shape=(32, 32, 1), name='encoder_input')
x = layers.Conv2D(32, (3,3), activation='relu', strides=2, padding='same')(encoder_input)
x = layers.Conv2D(64, (3,3), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(128, (3,3), activation='relu', strides=2, padding='same')(x)
sbf = K.int_shape(x)[1:]
x = layers.Flatten()(x)
mean = layers.Dense(12, name='mean')(x)
log_var = layers.Dense(12, name='log_var')(x)
z = utils.Sampling()([mean, log_var])
encoder = models.Model(encoder_input, [mean, log_var, z], name='encoder')
encoder.summary() # resumen del modelo 
decoder_input = layers.Input(shape=(12,), name='decoder_input') # Dimensiones de entrada (n,) para la dimensión latente
# Añadir capas densas y de reestructuración
x = layers.Dense(np.prod(sbf), activation='relu')(decoder_input) # Reestructurar a la forma antes de aplanar
x = layers.Reshape(sbf)(x) # Volver a dar forma a la salida
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same',strides = 2)(x) # Capas deconvolucionales
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same',strides = 2)(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same',strides = 2)(x)
# Definir la capa de salida del decodificador
decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x) # Salida de 1 canal (escala de grises)
# Definir el modelo del decodificador
decoder = models.Model(decoder_input, decoder_output, name='decoder')
decoder.summary() # resumen del modelo

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data() # Cargar el conjunto de datos Fashion MNIST
x_train = utils.preprocess(x_train) # Preprocesar las imágenes de entrenamiento
x_test = utils.preprocess(x_test) # Preprocesar las imágenes de prueba

vae = utils.VAE(encoder, decoder) # Crear el modelo VAE
vae.compile(optimizer=tf.keras.optimizers.Adam()) # Compilar el modelo
vae.fit(x_train, epochs=5, batch_size=64) # Entrenar el modelo
example_images = x_test[:10] # Seleccionar 10 imágenes de prueba para la visualización
predictions = vae.predict(example_images)[:][2] # Hacer predicciones con el auto-codificador variacional
# Visualizar las imágenes originales y reconstruidas
utils.visualize(example_images, predictions, 10) # Visualizar las imágenes originales y reconstruidas