# Modelo de Machine Learning

## Descripción General
Este script implementa un modelo de machine learning que realiza tareas de clasificación o regresión, como la carga de datos, preprocesamiento, entrenamiento y evaluación del modelo.

## Librerías
Este script utiliza las siguientes bibliotecas:
- `pandas`: Para la manipulación y análisis de datos.
- `sklearn`: Para los algoritmos de machine learning, división de datos y métricas de evaluación.

## Paso 1: Descripción del Código
En este paso, se importan las bibliotecas necesarias para machine learning y procesamiento de datos.

### Código
```python
from IPython.display import IFrame

# Insertar un PDF usando un IFrame
file_path = "Proyecto.pdf" 
IFrame(file_path, width=900, height=700)
```

## Paso 2: Descripción del Código
En este paso, se importan las bibliotecas necesarias para machine learning y procesamiento de datos.

### Código
```python
# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
```

## Paso 3: Descripción del Código
Se carga el conjunto de datos desde un archivo CSV a un DataFrame usando pandas.

### Código
```python
# Cargar el dataset
doc = "diabetes.csv"
df = pd.read_csv(doc)
```

## Paso 4: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Mostrar las primeras filas del dataset
print(df.head())
```

## Paso 5: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Descripción estadística básica
print(df.describe())
```

## Paso 6: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Evaluar las columnas
print(df.columns)
```

## Paso 7: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Listado de columnas que no deberían tener ceros
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Reemplazar valores 0 con NaN en las columnas relevantes
for column in columns_with_zeros:
    df[column] = df[column].mask(df[column] == 0)

```

## Paso 8: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Verificar valores faltantes después de reemplazar ceros
missing_values_after_replacement = df.isnull().sum()
print("Valores faltantes después del reemplazo de ceros:")
print(missing_values_after_replacement)
```

## Paso 9: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Visualizar la distribución de las clases en la columna 'Outcome'
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=df)
plt.title("Distribución de las clases en 'Outcome' (0: No diabetes, 1: Diabetes)")
plt.xlabel("Clase (Outcome)")
plt.ylabel("Cantidad")
plt.show()
```

## Paso 10: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Gráfico de pares para visualizar la relación entre características y Outcomes
sns.pairplot(df, hue="Outcome")
plt.show()
```

## Paso 11: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Boxplot para analizar la variabilidad
plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.title("Boxplot de las características del dataset Iris")
plt.show()
```

## Paso 12: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Crear el mapa de calor para la matriz de correlación
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de correlación entre características numéricas")
plt.show()
```

## Paso 13: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Preprocesamiento: escalamiento de las características
features = df.drop(columns='Outcome')  
features = features.fillna(features.mean())  # Imputación de valores faltantes
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Aplicar PCA sin reducir el número de componentes (se obtienen todas las componentes)
pca = PCA()
pca.fit(scaled_features)

# Varianza explicada por cada componente
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Graficar la varianza explicada acumulada
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance)+1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada por Componentes Principales')
plt.grid(True)
plt.show()

```

## Paso 14: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Preprocesamiento: escalamiento de las características
features = df.drop(columns='Outcome')  
features = features.fillna(features.mean())  # Imputación de valores faltantes
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

## Paso 15: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Aplicar PCA con 4 componentes principales
pca_4 = PCA(n_components=4)
pca_4_components = pca_4.fit_transform(scaled_features)
```

## Paso 16: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Varianza explicada por las 4 componentes principales
explained_variance_ratio_4 = pca_4.explained_variance_ratio_
```

## Paso 17: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Mostrar la varianza explicada por cada componente y la suma total
print("Varianza explicada por cada componente:")
print(explained_variance_ratio_4)
print("Suma de varianza explicada por las 4 componentes:", np.sum(explained_variance_ratio_4))
```

## Paso 18: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Gráfico de barras para mostrar la varianza explicada por cada componente
plt.figure(figsize=(8, 6))
plt.bar(range(1, 5), explained_variance_ratio_4, tick_label=["PC1", "PC2", "PC3", "PC4"])
plt.xlabel('Componentes Principales')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Cada Componente Principal')
plt.show(
```

## Paso 19: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Crear una figura en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar las primeras tres componentes principales
scatter = ax.scatter(pca_4_components[:, 0], pca_4_components[:, 1], pca_4_components[:, 2],
                     c=pca_4_components[:, 3], cmap='viridis', s=60)

# Etiquetas de los ejes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Agregar una barra de colores para la cuarta componente
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('PC4')

plt.title('PCA con 4 Componentes Principales (PC1, PC2, PC3 y PC4)')
plt.show()

```

## Paso 20: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Preprocesamiento: escalamiento de las características
features = df.drop(columns='Outcome')  
features = features.fillna(features.mean())  # Imputación de valores faltantes
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

## Paso 21: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Aplicar PCA con 4 componentes principales
pca_4 = PCA(n_components=4)
pca_4.fit(scaled_features)
pca_features = pca.fit_transform(scaled_features)
```

## Paso 22: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Obtener la matriz de cargas
loadings = pd.DataFrame(pca_4.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=features.columns)
```

## Paso 23: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Mostrar las cargas de cada característica en los 4 componentes principales
print(loadings)
```

## Paso 24: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Preprocesamiento: escalamiento de las características
features = df.drop(columns='Outcome')  
labels = df['Outcome']  # Esta es la variable objetivo
features = features.fillna(features.mean())  # Imputar valores faltantes con la media
```

## Paso 25: Descripción del Código
Se divide el conjunto de datos en entrenamiento y prueba para evaluar el rendimiento del modelo.

### Código
```python
# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.2, random_state=42)
```

## Paso 26: Descripción del Código
Se entrena el modelo de machine learning utilizando los datos de entrenamiento.

### Código
```python
# Entrenar un modelo de Regresión Logística
model = LogisticRegression()
model.fit(X_train, y_train)
```

## Paso 27: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)
```

## Paso 28: Descripción del Código
Se evalúa la precisión del modelo comparando las predicciones con los valores reales.

### Código
```python
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

## Paso 29: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
```

## Paso 30: Descripción del Código
En este paso, se importan las bibliotecas necesarias para machine learning y procesamiento de datos.

### Código
```python
from sklearn.ensemble import RandomForestClassifier

# Entrenar un modelo de Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

```

## Paso 31: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)
```

## Paso 32: Descripción del Código
Se evalúa la precisión del modelo comparando las predicciones con los valores reales.

### Código
```python
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

## Paso 33: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
```

## Paso 34: Descripción del Código
Se entrena el modelo de machine learning utilizando los datos de entrenamiento.

### Código
```python
# Entrenar un modelo de Support Vector Machine (SVM)
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)
```

## Paso 35: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)
```

## Paso 36: Descripción del Código
Se evalúa la precisión del modelo comparando las predicciones con los valores reales.

### Código
```python
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

## Paso 37: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
```

## Paso 38: Descripción del Código
Se entrena el modelo de machine learning utilizando los datos de entrenamiento.

### Código
```python
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)
```

## Paso 39: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
report = classification_report(y_test, y_pred)
```

## Paso 40: Descripción del Código
Se evalúa la precisión del modelo comparando las predicciones con los valores reales.

### Código
```python
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

## Paso 41: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
```

## Paso 42: Descripción del Código
Se divide el conjunto de datos en entrenamiento y prueba para evaluar el rendimiento del modelo.

### Código
```python
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.2, random_state=42)

# Entrenar el modelo de Regresión Logística
model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)

# Entrenar el modelo de Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Entrenar el modelo SVM
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

```

## Paso 43: Descripción del Código
Se evalúa la precisión del modelo comparando las predicciones con los valores reales.

### Código
```python
# Los valores de prediccion
input_data = np.array([[10,	168,	74,	0	,0	,38,	0.537,	34]])

# Preprocesar el input_data
scaled_input = scaler.transform(input_data)  # Escalar usando el mismo scaler

# Aplicar PCA
pca_input = pca.transform(scaled_input)  # Aplicar el mismo PCA

# Hacer las predicciones y calcular la precisión
logistic_pred = model_logistic.predict(pca_input)
logistic_accuracy = accuracy_score(y_test, model_logistic.predict(X_test))  # Accuracy en el conjunto de prueba

rf_pred = model_rf.predict(pca_input)
rf_accuracy = accuracy_score(y_test, model_rf.predict(X_test))  # Accuracy en el conjunto de prueba

svm_pred = model_svm.predict(pca_input)
svm_accuracy = accuracy_score(y_test, model_svm.predict(X_test))  # Accuracy en el conjunto de prueba

nb_pred = model_nb.predict(pca_input)
nb_accuracy = accuracy_score(y_test, model_nb.predict(X_test))

mlp_pred = model_mlp.predict(pca_input)
mlp_accuracy = accuracy_score(y_test, model_mlp.predict(X_test))

# Crear la tabla con las predicciones y la precisión de todos los modelos
results = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Random Forest', 'SVM', 'Naive Bayes', 'Redes Neuronales (MLP)'],
    'Predicción': [
        'Tiene diabetes' if logistic_pred[0] == 1 else 'No tiene diabetes',
        'Tiene diabetes' if rf_pred[0] == 1 else 'No tiene diabetes',
        'Tiene diabetes' if svm_pred[0] == 1 else 'No tiene diabetes',
        'Tiene diabetes' if nb_pred[0] == 1 else 'No tiene diabetes',
        'Tiene diabetes' if mlp_pred[0] == 1 else 'No tiene diabetes'
    ],
    'Precisión (%)': [
        logistic_accuracy * 100,
        rf_accuracy * 100,
        svm_accuracy * 100,
        nb_accuracy * 100,
        mlp_accuracy * 100
    ]
})

# Mostrar la tabla
print(results)
```

## Paso 44: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python
results
```

## Paso 45: Descripción del Código
Este bloque realiza operaciones como preprocesamiento, selección de características, o evaluación del modelo.

### Código
```python

```

## Resultados
El script genera predicciones para el conjunto de prueba y evalúa su rendimiento utilizando métricas como la precisión. Los resultados pueden variar según los datos y el tipo de modelo empleado.

