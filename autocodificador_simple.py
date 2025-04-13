from tensorflow.keras import datasets
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data() # Cargar el conjunto de datos Fashion MNIST

def preprocess(imgs):
    imgs = imgs.astype('float32') / 255.0 # Normalizar a 0-1
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0) # Añadir padding de 2 píxeles
    imgs = np.expand_dims(imgs, axis=-1) # Añadir dimensión de canal
    return imgs

def visualize(samples,predictions,n):
    # Visualizar las imágenes originales y reconstruidas
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 10) # Crear una cuadrícula de subgráficas
    for i in range(n):
        # Mostrar la imagen original
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(samples[i].reshape(32, 32), cmap='gray')
        ax.axis('off')
        # Mostrar la imagen reconstruida
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(predictions[i].reshape(32, 32), cmap='gray')
        ax.axis('off')
    plt.show() # Mostrar la figura

def makeModel(nlyr, latent_dim):
    # Definir el codificador
    encoder_input = layers.Input(shape=(32, 32, 1), name='encoder_input') # Dimensiones de entrada (32, 32, 1) para imágenes de 32x32 píxeles con 1 canal (escala de grises)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides = 2)(encoder_input) # Capas convolucionales
    n = 6 # 2^5 = 32, el tamaño de la imagen de entrada
    for i in range(nlyr-1):
        # Añadir capas convolucionales y de agrupamiento
        x = layers.Conv2D(pow(2,n), (3, 3), activation='relu', padding='same',strides = 2)(x) # Capas convolucionales
        n += 1
    before_flattening = K.int_shape(x)[1:]
    x = layers.Flatten()(x)
    # Definir la capa densa de salida del codificador
    encoder_output = layers.Dense(latent_dim, name='encoder_output')(x) # n dimensiones para la dimensión latente
    encoder = models.Model(encoder_input, encoder_output, name='encoder') # Definir el modelo del codificador
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input') # Dimensiones de entrada (n,) para la dimensión latente
    # Añadir capas densas y de reestructuración
    x = layers.Dense(np.prod(before_flattening), activation='relu')(decoder_input) # Reestructurar a la forma antes de aplanar
    x = layers.Reshape(before_flattening)(x) # Volver a dar forma a la salida
    n = 5 +nlyr-1 # 2^(5+lyr) => tamaño de la imagen de entrada
    for i in range(nlyr): 
        x = layers.Conv2DTranspose(pow(2,n), (3, 3), activation='relu', padding='same',strides = 2)(x) # Capas deconvolucionales
        n -= 1
    # Definir la capa de salida del decodificador
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x) # Salida de 1 canal (escala de grises)
    # Definir el modelo del decodificador
    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    encoder.summary() # Resumen del modelo del codificador
    decoder.summary() # Resumen del modelo del decodificador
    # Definir el modelo del auto-codificador variacional
    autoencoder = models.Model(encoder_input, decoder(encoder_output), name='autoencoder')
    # Compilar el modelo
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # Usar binary_crossentropy como función de pérdida
    return autoencoder,encoder,decoder # Devolver el modelo del auto-codificador variacional, el codificador y el decodificador

x_train = preprocess(x_train) # Preprocesar los datos de entrenamiento
x_test = preprocess(x_test) # Preprocesar los datos de prueba
# Entrenar el modelo
autoencoder,encoder,decoder = makeModel(4, 128)
autoencoder.fit(x_train, x_train, epochs=5, batch_size=64,shuffle = True, validation_split = .2) # Entrenar el auto-codificador variacional
example_images = x_test[:10] # Seleccionar 10 imágenes de prueba para la visualización
predictions = autoencoder.predict(example_images) # Hacer predicciones con el auto-codificador variacional
# Visualizar las imágenes originales y reconstruidas
visualize(example_images, predictions, 10) # Visualizar las imágenes originales y reconstruidas

# Obtener los valores mínimo y máximo de los embeddings
embeddings = encoder.predict(x_test)
mins, maxs = np.min(embeddings), np.max(embeddings)
sample = np.random.uniform(mins, maxs, (18, 128)) # Generar muestras aleatorias en el espacio latente
reconstructions = decoder.predict(sample) # Hacer predicciones con el decodificador
# Visualizar las reconstrucciones de las muestras aleatorias    
visualize(reconstructions, reconstructions, 10) # Visualizar las reconstrucciones de las muestras aleatorias
