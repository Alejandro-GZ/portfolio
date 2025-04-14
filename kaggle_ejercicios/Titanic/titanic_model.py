import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#Leer archivos csv
file_path = 'titanic_data\\train.csv'  
data = pd.read_csv(file_path)
file_path = 'titanic_data\\test.csv'  
data2 = pd.read_csv(file_path)

def preprocess(data):
    ftrs = ["Pclass","Sex","SibSp","Parch","Embarked"]
    x = pd.get_dummies(data[ftrs])
    x["LessThan1"] = data["Age"].apply(lambda x: 1 if x < 1 else 0)
    x["LessThan10"] = data["Age"].apply(lambda x: 1 if x < 10 else 0)
    x["OlderThan40"] = data["Age"].apply(lambda x: 1 if x > 40 else 0)
    x["OlderThan60"] = data["Age"].apply(lambda x: 1 if x > 60 else 0)
    x["GreaterFareThan100"] = data["Fare"].apply(lambda x: 1 if x > 100 else 0)
    x["GreaterFareThan50"] = data["Fare"].apply(lambda x: 1 if x > 50 else 0)
    return x
x = preprocess(data)
y = data["Survived"]
# Crear y entrenar el modelo
model = RandomForestClassifier(n_estimators=1000, max_depth=10)
model.fit(x, y)
print(model.score(x, y))
# Predecir sobre el conjunto de test
x_test = preprocess(data2)
data2["Survived"] = model.predict(x_test)
#Escribir resultados en un archivo csv junto a PassengerId
dataR = data2[["PassengerId", "Survived"]]
dataR.to_csv('titanic_data\\submission.csv', index=False)


