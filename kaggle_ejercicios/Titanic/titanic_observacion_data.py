import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV en un DataFrame de pandas
file_path = 'titanic_data\\train.csv'  
data = pd.read_csv(file_path)

# Gráfico según el sexo de los pasajeros y su supervivencia -> Sexo relevante, las mujeres sobrevivieron más que los hombres
sns.countplot(data=data, x='Survived', hue='Sex', palette=['blue','red'])

plt.xlabel('Sobrevivió (0 = No, 1 = Sí)')
plt.ylabel('Cantidad de Personas')
plt.title('Supervivencia según Sexo')
plt.legend(title='Sexo', loc='upper right', labels=['Hombre', 'Mujer'])

plt.show()

# Gráfico según la clase de los pasajeros y su supervivencia -> Clase relevante, la 1ra clase sobrevivió más que la 2da y 3ra
sns.countplot(data=data, x='Survived', hue='Pclass', palette=['blue','red','green'])

plt.xlabel('Sobrevivió (0 = No, 1 = Sí)')
plt.ylabel('Cantidad de Personas')
plt.title('Supervivencia según su Clase')
plt.legend(title='Clase', loc='upper right', labels=['Alta', 'Media', 'Baja'])

plt.show()

# Histograma de la edad de los pasajeros que sobrevivieron y no sobrevivieron -> Edad relevante, importante destacar el aumento de supervivencia en pasajerso de entre 0 y 10 años
sns.histplot(data=data, x='Age', hue='Survived', bins=30, kde=True, palette=['grey','green'])

plt.xlabel('Edad')
plt.ylabel('Cantidad de Personas')
plt.title('Histograma de Edad según Supervivencia')

plt.show()

# Histograma de supervivencia según la tarifa pagada -> Tarifa relevante, a mayor tarifa, mayor probabilidad de supervivencia
sns.histplot(data=data, x='Fare', hue='Survived', palette=['grey','green'])

plt.xlabel('Tarifa Pagada')
plt.ylabel('Cantidad de Personas')  
plt.title('Histograma de Tarifa según Supervivencia')

plt.show()

# Gráfico de supervivencia según numero de hermanos / cónyuges a bordo -> Relevante
sns.countplot(data=data, x='SibSp', hue='Survived', palette=['grey','green'])

plt.xlabel('Número de Hermanos/Cónyuges a Bordo')
plt.ylabel('Cantidad de Personas')
plt.title('Supervivencia según Hermanos/Cónyuges a Bordo')

plt.show()

# Gráfico de supervivencia según numero de padres / hijos a bordo -> Relevante
sns.countplot(data=data, x='Parch', hue='Survived', palette=['grey','green'])

plt.xlabel('Número de Padres/Hijos a Bordo')
plt.ylabel('Cantidad de Personas')
plt.title('Supervivencia según Padres/Hijos a Bordo')

plt.show()

# Gráfico de supervivencia según puerto de embarque -> Relevante, el puerto de embarque influye en la probabilidad de supervivencia
sns.countplot(data=data, x='Embarked', hue='Survived', palette=['grey','green'])

plt.xlabel('Puerto de Embarque')
plt.ylabel('Cantidad de Personas')
plt.title('Supervivencia según Puerto de Embarque')

plt.show()

# Gráfico de supervivencia de pasajeros menores de 1 año -> Relevante, todos los pasajeros menores de 1 año sobrevivieron
sns.histplot(data[data['Age'] < 1], x='Age', hue='Survived', bins=30, kde=True, palette=['grey','green'])

plt.xlabel('Edad')
plt.ylabel('Cantidad de Personas')
plt.title('Histograma de Edad según Supervivencia (Menores de 1 año)')

plt.show()
