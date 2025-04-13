from tensorflow.keras import datasets
import numpy as np
import autocodificador_utils as utils

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data() # Cargar el conjunto de datos Fashion MNIST

x_train = utils.preprocess(x_train) # Preprocesar los datos de entrenamiento
x_test = utils.preprocess(x_test) # Preprocesar los datos de prueba
# Entrenar el modelo
autoencoder,encoder,decoder = utils.makeModelSimple(4, 128)
autoencoder.fit(x_train, x_train, epochs=5, batch_size=64,shuffle = True, validation_split = .2) # Entrenar el auto-codificador variacional
example_images = x_test[:10] # Seleccionar 10 imágenes de prueba para la visualización
predictions = autoencoder.predict(example_images) # Hacer predicciones con el auto-codificador variacional
# Visualizar las imágenes originales y reconstruidas
utils.visualize(example_images, predictions, 10) # Visualizar las imágenes originales y reconstruidas

# Obtener los valores mínimo y máximo de los embeddings
embeddings = encoder.predict(x_test)
mins, maxs = np.min(embeddings), np.max(embeddings)
sample = np.random.uniform(mins, maxs, (18, 128)) # Generar muestras aleatorias en el espacio latente
reconstructions = decoder.predict(sample) # Hacer predicciones con el decodificador
# Visualizar las reconstrucciones de las muestras aleatorias    
utils.visualize(reconstructions, reconstructions, 10) # Visualizar las reconstrucciones de las muestras aleatorias
