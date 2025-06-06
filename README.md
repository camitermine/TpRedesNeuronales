# Trabajo Práctico - Redes Neuronales

Este proyecto se enfoca en el desarrollo de una red neuronal de clasificación para predecir el nivel de compromiso (EngagementLevel) de jugadores de videojuegos, utilizando el siguiente dataset: [Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)


## Parte 1: Análisis de la Base de Datos

### 1. Descripción de cada columna del conjunto de datos:
Columnas descartadas:

1. PlayerID: Identificador único por jugador. (Categórica)
2. Location: Ubicación geográfica del jugador. (Categórica)
3. InGamePurchases: Indica si realiza compras. (Discreta binaria)
4. PlayerLevel: Nivel actual en el juego. (Discreta)
5. AchievementsUnlocked: Cantidad de logros. (Discreta)

Columnas utilizadas para el entrenamiento:

1. Age: Edad del jugador. (Discreta)
2. Gender: Género del jugador. (Categórica nominal)
3. GameGenre: Género del juego. (Categórica nominal)
4. PlayTimeHours: Horas promedio jugadas. (Continua)
5. GameDifficulty: Nivel de dificultad. (Categórica ordinal)
6. SessionsPerWeek: Cantidad de sesiones semanales. (Discreta)
7. AvgSessionDurationMinutes: Duración promedio por sesión. (Continua)

Variable objetivo:

EngagementLevel: Nivel de compromiso (Low, Medium, High). (Categórica ordinal)

### 2. Análisis de Correlaciones:
![Figure_1](https://github.com/user-attachments/assets/b97234db-2a48-4d6b-b2f3-744a39d9b7f7)

Las variables que muestran una mayor correlación positiva con la columna objetivo (EngagementLevel) son:

SessionsPerWeek : 0.61

AvgSessionDurationMinutes : 0.48

Estas dos variables están relacionadas con el nivel de compromiso, lo cual tiene sentido, ya que quienes juegan más seguido y durante más tiempo por sesión tienden a estar más comprometidos.

En cambio, variables como:
PlayTimeHours, Age tienen una correlación prácticamente nula pero eso no implica que no esten relacionadas con el nivel de compromiso, sino que esa relacion no es lineal.

### 3. Análisis de Factibilidad:

La base de datos es adecuada para entrenar una red neuronal de clasificación, ya que contiene una columna objetivo categórica (EngagementLevel) y varias características numéricas que permiten establecer patrones.
El propósito del modelo es predecir el nivel de compromiso de los jugadores (bajo, medio o alto) en función de su comportamiento dentro del juego, como la cantidad de sesiones semanales y la duración promedio de cada sesión. Esto puede ser útil para personalizar experiencias de juego o diseñar estrategias de retención de los jugadores.

### 4. Datos Atípicos y Limpieza de Datos:

Se analizaron las columnas numéricas en busca de valores atípicos utilizando el método del rango intercuartílico. No se detectaron outliers significativos, ni valores nulos o duplicados en los datos.
Por lo tanto, no fue necesaria una limpieza adicional, ya que la base de datos se encontraba en muy buen estado para el entrenamiento del modelo.

### 5. Transformaciones Preliminares

Se realizaron transformaciones necesarias para preparar los datos antes del entrenamiento. 
Se codificó la variable objetivo `EngagementLevel` en valores numéricos (0: Bajo, 1: Medio, 2: Alto) 
La variable `EngagementLevel` también en valores numéricos (0: Fácil, 1: Medio, 2: Difícil)
Las variables categóricas nominales, GameGenre y Gender, se convirtieron utilizando One-Hot Encoding. 
Todas las variables numéricas se normalizaron utilizando MinMaxScaler. 

## Parte 2: Desarrollo de la Red Neuronal

### 1. Arquitectura de Red

![redNeuronal](https://github.com/user-attachments/assets/ca86fc50-922d-4675-92c4-47aa92483e72)

Capa de Entrada : Tiene 10 neuronas (una por cada columna caracteristica)
Capa Oculta : Inicialmente va a tener 16 neuronas (un numero no tan grande para evitar overfitting)
Capa de salida: Tiene 3 posibles resultados mi columna objetivo (bajo, medio y alto) asi que serán 3 neuronas

En la capa oculta se va a usar la funcion de activación ReLu para acelerar el entrenamiento y mitigar el problema del gradiente desvaneciente
En la capa de salida se utilizará la función de activación SoftMax ya que hay mas de dos salidas posibles (clasificación multiclase)

### 2. Implementación en Numpy
Se implementó toda la red usando solo NumPy. 
Incluye:
Inicialización aleatoria de pesos y sesgos
Activaciones: ReLU y Softmax
Cálculo del costo con cross-entropy
Retropropagación manual
Ajuste de pesos con descenso por gradiente estocástico (SGD puro)

### 3. Entrenamiento y Evaluación
![Figure_3](https://github.com/user-attachments/assets/4fae1102-5bfa-4ba8-9855-5bf6e8771467)
La red fue entrenada usando aproximadamente 32.000 registros, con una sola capa oculta y 100 épocas (epochs). 
Durante el entrenamiento se trazaron curvas de precisión y de costo.
Precisión obtenida:
Entrenamiento: 87.45%
Prueba: 87.85%

La evolución del entrenamiento se visualizó mediante:
Curva de pérdida (función de costo) por época
Curvas de precisión (accuracy) para los conjuntos de entrenamiento y prueba

Estas métricas muestran que el modelo tiene un buen desempeño y no presenta signos evidentes de sobreajuste.

### 4. Análisis de Overfitting

No se detectó sobreajuste:
Las precisiones en train y test son similares
El costo disminuye sin oscilaciones bruscas

## Parte 3: Comparación con scikit-learn
Mi red neuronal tardó mucho mas tiempo que la implementada con scikit-learn
En razgos generales las presiciones son muy similares y no hubo diferencias significativas
A continuación se muestra la matriz de confusión:

![Figure_4](https://github.com/user-attachments/assets/976fabf3-6c88-4941-8c6d-c86343f986c4)

## Parte 4: Conclusión Final
Durante el desarrollo de la red comprendí en profundidad cómo funciona una red neuronal desde sus fundamentos. 
Al construirla manualmente con NumPy, entendí paso a paso el proceso de entrenamiento: desde la inicialización de pesos, el cálculo de activaciones, la retropropagación, hasta la actualización con descenso por gradiente. 

Ventajas de implementar la red manualmente:
Aprendizaje profundo del funcionamiento interno de las redes.

Posibilidad de personalizar el comportamiento y entender mejor errores o comportamientos inesperados.

Desventajas:
Es más propenso a errores 
Menor eficiencia en entrenamiento y mucho mas lento.

Por otro lado, utilizar scikit-learn permite obtener resultados rápidos y confiables con muy poco código. Aunque limita el control sobre ciertos aspectos internos del modelo. No tenes tanta versatilidad.

