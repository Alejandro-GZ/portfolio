import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def makeModelSimple(nlyr, latent_dim):
    '''Crear un auto-codificador simple con capas convolucionales y deconvolucionales.'''
    '''nlyr: número de capas convolucionales y deconvolucionales. latent_dim: dimensión del espacio latente.'''
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

class Sampling(layers.Layer):
    """Usa la media y la desviación estándar para muestrear un punto en el espacio latente."""

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
class VAE(models.Model):
    """Modelo de Autoencoder Variacional."""

    def __init__(self, encoder, decoder,beta , **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
    def call(self, inputs):
        mean, log_var, z = self.encoder(inputs)
        reconstrucion = self.decoder(z)
        return mean, log_var, reconstrucion
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, log_var, reconstrucion = self(data)
            reconstruction_loss = tf.reduce_mean(
                self.beta 
                * tf.keras.losses.binary_crossentropy(data, reconstrucion, axis=(1, 2, 3))
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 
                    * (1 + log_var - tf.square(mean) - tf.exp(log_var)),
                    axis = 1,)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return { m.name: m.result() for m in self.metrics }
    
def preprocess(imgs):
    '''Preprocesar las imágenes de entrada para el auto-codificador variacional.'''
    '''imgs: imágenes de entrada. Devuelve las imágenes preprocesadas.'''
    imgs = imgs.astype('float32') / 255.0 # Normalizar a 0-1
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0) # Añadir padding de 2 píxeles
    imgs = np.expand_dims(imgs, axis=-1) # Añadir dimensión de canal
    return imgs

def visualize(samples,predictions,n):
    '''Visualizar las imágenes originales y reconstruidas.'''
    '''samples: imágenes originales. predictions: imágenes reconstruidas. n: número de imágenes a visualizar.'''
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
