#cargar el dataset de entrenamiento
import pandas as pd

data = pd.read_csv("data\\train.csv")
data2 = pd.read_csv("data\\test.csv")
    
# propiedades relevantes filtradas según su tipo de dato (numerico o categorico)
numeric_features = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']
categorical_features = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment']
#NOTA: las demás propiedades son descartadas debido a que no son relevantes para el entrenamiento o no es necesaria/posible su reparación

# Funcion de perdida rmse
def rmse(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras.losses import MeanSquaredError
    mse = MeanSquaredError()
    return tf.sqrt(mse(y_true, y_pred))

#preprocesar los datos
def preprocesar(data, is_test=False):
    # Eliminar filas con valores nulos en la columna 'Listening_Time_minutes'
    if not is_test :
        data = data.dropna(subset=['Listening_Time_minutes'])
    # Rellenar nulos en la columna id con el valor de su index
    data['id'] = data['id'].fillna(pd.Series(data.index.astype(int), index=data.index))
    # Rellenar nulos con la media de las filas con mismo nombre de podcast
    for ft in numeric_features:
        data[ft] = data[ft].fillna(data.groupby('Podcast_Name')[ft].transform('mean'))
    # Rellenar nulos con la moda de las filas con mismo nombre de podcast
    for ft in categorical_features:
        data[ft] = data[ft].fillna(data.groupby('Podcast_Name')[ft].transform(lambda x: x.mode()[0]))
    # Estandarizar los datos numericos
    data[numeric_features] = (data[numeric_features] - data[numeric_features].mean()) / data[numeric_features].std()
    # Crear dummies para las variables categoricas
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    # Eliminar columnas innecesarias
    data = data.drop(columns=['id','Podcast_Name','Episode_Title'])
    data = data.dropna()
    return data
def fitAndPredict(model, X_train, y_train, X_test):
    categorical_model = RandomForestRegressor(n_estimators=20,max_depth=20)
    categorical_model.fit(X_train.drop(columns=numeric_features).astype('float64'), y_train) # las primeras 5 columnas no son categoricas
    y_train_cat = categorical_model.predict(X_train.drop(columns=numeric_features).astype('float64'))
    y_train_cat = ((y_train_cat - y_train_cat.mean()) / y_train_cat.std()).astype('float64')
    nf = numeric_features + ['TreeDecision']
    X_train["TreeDecision"] = y_train_cat
    X_train_nn = X_train[nf]
    X_train_nn = X_train_nn.to_numpy().astype('float64')
    # entrenar el modelo
    model.fit(X_train_nn, y_train, epochs=100, batch_size=512)
    y_test_cat = categorical_model.predict(X_test.drop(columns=numeric_features).astype('float64'))
    y_test_cat = ((y_test_cat - y_test_cat.mean()) / y_test_cat.std()).astype('float64')
    X_test["TreeDecision"] = y_test_cat
    X_test_nn = X_test[nf]
    # escribir el resultado a un archivo csv junto con el id
    data2['Listening_Time_minutes'] = model.predict(X_test_nn, batch_size=16)
    result = pd.DataFrame(data2, columns=['id','Listening_Time_minutes'])
    #guardar el resultado en un archivo csv
    result.to_csv('submission.csv', index=False)


# preprocesar los datos de entrenamiento
data = preprocesar(data)

# Fase de testing de modelo:

# separar las variables dependientes e independientes
X_train = data.drop(columns=['Listening_Time_minutes'])
y_train = data['Listening_Time_minutes'].to_numpy().astype('float64')
X_test = preprocesar(data2, True)

# entrenar el modelo: IDEA RandomForestRegressor para las categoricas y un modelo de red neuronal para las numericas
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Input(shape=(5,)))
model.add(layers.Dense(128, activation='leaky_relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(256, activation='leaky_relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='leaky_relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1024, activation='leaky_relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='leaky_relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(256, activation='leaky_relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='leaky_relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1, activation='gelu'))
model.compile(optimizer='sgd', loss=rmse)
model.summary()

fitAndPredict(model, X_train, y_train, X_test)
# guardar el modelo
model.save('model.h5')