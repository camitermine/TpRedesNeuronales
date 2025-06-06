#Trabajo Práctico Matemática III - Redes Neuronales

Este proyecto se enfoca en el desarrollo de una red neuronal de clasificación para predecir el nivel de compromiso (EngagementLevel) de jugadores de videojuegos, utilizando el siguiente dataset: [Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)


##Parte 1: Análisis de la Base de Datos

1. Descripción de cada columna del conjunto de datos 

**No se van a utilizar para entrenar la red**

PlayerID: 
Representación: Identificador único para cada jugador.
Tipo de Variable: Categórica. 

Location (Ubicación):
Representación: Ubicación geográfica del jugador.
Tipo de Variable: Categórica. 

InGamePurchases (Compras Dentro del Juego):
Representación: Indica si el jugador realiza compras (1=Si, 0=No).
Tipo de Variable: Discreta (binaria).

PlayerLevel (Nivel del Jugador):
Representación: Nivel actual del jugador en el juego.
Tipo de Variable: Discreta.

AchievementsUnlocked (Logros Desbloqueados):
Representación: Número total de logros desbloqueados.
Tipo de Variable: Discreta.

**Son las columnas elegidas**
Age (Edad):
Representación: Edad del jugador.
Tipo de Variable: Discreta.

Gender (Género):
Representación: Género del jugador ('Male', 'Female').
Tipo de Variable: Categórica. 

GameGenre (Género del Juego):
Representación: Género del videojuego ('RPG', 'Action', 'Strategy', etc.).
Tipo de Variable: Categórica. 

PlayTimeHours (Horas de Juego):
Representación: Promedio de horas jugadas por sesión.
Tipo de Variable: Continua.

GameDifficulty (Dificultad del Juego):
Representación: Nivel de dificultad del juego ('Easy', 'Medium', 'Hard').
Tipo de Variable: Categórica Ordinal (tienen jerarquia esas categorias). 

SessionsPerWeek (Sesiones por Semana):
Representación: Número de sesiones de juego por semana.
Tipo de Variable: Discreta.

AvgSessionDurationMinutes (Duración Promedio de Sesión en Minutos):
Representación: Duración promedio de cada sesión en minutos.
Tipo de Variable: Continua.

**Es mi columna objetivo**
EngagementLevel (Nivel de Compromiso):
Representación: Variable objetivo: 'High', 'Medium', o 'Low' compromiso.
Tipo de Variable: Categórica Ordinal (tienen jerarquia esas categorias). 

