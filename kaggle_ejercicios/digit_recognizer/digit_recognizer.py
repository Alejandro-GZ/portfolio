#Cargar csv a DataFrame de pandas
import pandas as pd
data = pd.read_csv('digit_recognizer_data\\train.csv')
data_test = pd.read_csv('digit_recognizer_data\\test.csv')
#Separar en x e y
y = data['label']
x = data.drop(columns=['label'])
#Normalizar los datos
x = x / 255.0

#Nota: imagenes de 28x28 = 784 pixeles
#Redimensionar los datos
x = x.values.reshape(-1, 28, 28, 1)
#CNN
from keras import layers, models
model = models.Sequential()
model.add(layers.Input(shape=(28,28,1))) #Input layer
model.add(layers.Conv2D(72,(4,4),activation='relu',padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(72,(4,4),activation='relu',padding='same'))
model.add(layers.AveragePooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(layers.AveragePooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.AveragePooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
#Entrenar el modelo
model.fit(x,y,epochs=20,batch_size=64)
#Evaluar todas las imagenes de test y escribir el resultado a un csv
import numpy as np
x_test = data_test.values.reshape(-1, 28, 28, 1) / 255.0
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
#Guardar el resultado en un csv
submission = pd.DataFrame({'ImageId': np.arange(1, len(y_pred) + 1), 'Label': y_pred})
submission.to_csv('digit_recognizer_data\\submission.csv', index=False)
