import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


ruta = "online_gaming_behavior_dataset.csv"
df = pd.read_csv(ruta)

# 1. Selección de columnas relevantes
columnas = [
    'Age',
    'Gender',
    'GameGenre',
    'PlayTimeHours',
    'GameDifficulty',
    'SessionsPerWeek',
    'AvgSessionDurationMinutes',
    'EngagementLevel'
]
df_inicial = df[columnas].copy()

# 2. Limpieza de nulos y duplicados
df_inicial.dropna(inplace=True)
df_inicial.drop_duplicates(inplace=True)

# 3. Codificación de variables categóricas y la variable objetivo
mapeo_engagement = {'Low': 0, 'Medium': 1, 'High': 2}
df_inicial['EngagementLevel'] = df_inicial['EngagementLevel'].map(mapeo_engagement)

mapeo_dificultad = {'Easy': 0, 'Medium': 1, 'Hard': 2}
df_inicial['GameDifficulty'] = df_inicial['GameDifficulty'].map(mapeo_dificultad)

df_procesado = pd.get_dummies(df_inicial, columns=['GameGenre', 'Gender'], drop_first=True)

# Separar características (X) y variable objetivo (y)
X = df_procesado.drop('EngagementLevel', axis=1)
y = df_procesado['EngagementLevel']

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Tratamiento de valores atípicos con IQR
columnas_numericas_continuas = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes']
for col in columnas_numericas_continuas:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    minimo = Q1 - 1.5 * IQR
    maximo = Q3 + 1.5 * IQR

    X_train[col] = np.where(X_train[col] < minimo, minimo, X_train[col])
    X_train[col] = np.where(X_train[col] > maximo, maximo, X_train[col])

    X_test[col] = np.where(X_test[col] < minimo, minimo, X_test[col])
    X_test[col] = np.where(X_test[col] > maximo, maximo, X_test[col])

# 6. Normalización de variables numéricas
escalador = MinMaxScaler()
X_train_scaled = escalador.fit_transform(X_train)
X_test_scaled = escalador.transform(X_test)

X_train_np = X_train_scaled.T  
X_test_np = X_test_scaled.T   

encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = encoder.fit_transform(y_train.values.reshape(-1, 1)).T 
y_test_one_hot = encoder.transform(y_test.values.reshape(-1, 1)).T     


# --- Funciones de Activación ---

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def relu_backward(dA, Z):
    dZ = dA * (Z > 0) 
    return dZ

def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1)) # Sesgos son vectores columna
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    # Guardar los valores intermedios en un diccionario (cache) para usarlos en la retropropagación
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# --- 4. Función de Costo (Entropía Cruzada ya que se usa softmax) ---

def compute_cost(A2, Y):
    m = Y.shape[1] 

    cost = - (1 / m) * np.sum(Y * np.log(A2 + 1e-8))

    cost = np.squeeze(cost)
    return cost

# --- 5. Algoritmo de Retropropagación (Backpropagation) ---

def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1] 

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    Z1 = cache["Z1"]
    A2 = cache["A2"] # Activación de la capa de salida (predicciones)

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Gradientes de la Capa Oculta
    # Propagar el gradiente hacia atrás a través de W2
    dA1 = np.dot(W2.T, dZ2)
    # Aplicar la derivada de ReLU (relu_backward)
    dZ1 = relu_backward(dA1, Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2
    }
    return gradients

# --- 6. Ajuste de los Pesos (Descenso por Gradiente) ---

def update_parameters(parameters, gradients, learning_rate):
    """
    Actualiza los pesos y sesgos usando el descenso por gradiente.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    # Aplicar las actualizaciones (W = W - learning_rate * dW)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2
    }
    return parameters

# --- 7. Función Principal de Entrenamiento (nn_model) ---

def nn_model_sgd(X, Y, n_h, num_epochs=100, learning_rate=0.01, print_cost=True):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    m = X.shape[1]

    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        epoch_cost = 0

        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        for i in range(m):
            xi = X_shuffled[:, i].reshape(-1, 1)
            yi = Y_shuffled[:, i].reshape(-1, 1)

            A2, cache = forward_propagation(xi, parameters)
            gradients = backward_propagation(xi, yi, parameters, cache)
            parameters = update_parameters(parameters, gradients, learning_rate)

            epoch_cost += compute_cost(A2, yi)

        epoch_cost /= m
        costs.append(epoch_cost)

        # Precisión sobre todo el conjunto
        train_preds = predict(X, parameters)
        train_labels = np.argmax(Y, axis=0)
        train_accuracy = np.mean(train_preds == train_labels) * 100
        train_accuracies.append(train_accuracy)

        test_preds = predict(X_test_np, parameters)
        test_labels = np.argmax(y_test_one_hot, axis=0)
        test_accuracy = np.mean(test_preds == test_labels) * 100
        test_accuracies.append(test_accuracy)

        if print_cost and epoch % 10 == 0:
            print(f"Época {epoch} | Costo: {epoch_cost:.4f} | Train acc: {train_accuracy:.2f}% | Test acc: {test_accuracy:.2f}%")

    return parameters, costs, train_accuracies, test_accuracies


# --- 8. Función de Predicción ---

def predict(X, parameters):
    """
    Realiza predicciones utilizando los parámetros aprendidos.
    """
    A2, _ = forward_propagation(X, parameters)
    # np.argmax devuelve el índice de la probabilidad más alta, que es la clase predicha
    predictions = np.argmax(A2, axis=0)
    return predictions

# --- 9. Configuración y Ejecución del Entrenamiento ---
hidden_size = 16          
num_epochs = 100  

# # Entrenar el modelo

parameters, costs, train_accuracies, test_accuracies = nn_model_sgd(
    X_train_np, y_train_one_hot, hidden_size,
    num_epochs=num_epochs, learning_rate=0.01, print_cost=True
)


# # 10. Evaluación del Modelo 

# # Precisión en el conjunto de entrenamiento
train_predictions = predict(X_train_np, parameters)

# # Convertir las etiquetas one-hot verdaderas a etiquetas de clase para comparar
y_train_labels = np.argmax(y_train_one_hot, axis=0)
train_accuracy = np.mean(train_predictions == y_train_labels) * 100
print(f"Precisión en el conjunto de entrenamiento: {train_accuracy:.2f}%")

# # Precisión en el conjunto de prueba
test_predictions = predict(X_test_np, parameters)
y_test_labels = np.argmax(y_test_one_hot, axis=0)
test_accuracy = np.mean(test_predictions == y_test_labels) * 100
print(f"Precisión en el conjunto de prueba: {test_accuracy:.2f}%")

# # 11. Gráficos de Evaluación

plt.figure(figsize=(12, 5))

# # Costo
plt.subplot(1, 2, 1)
plt.plot(costs, label='Costo')
plt.title("Costo por Época")
plt.xlabel("Épocas")
plt.ylabel("Costo")
plt.grid(True)

# # Precisión
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Entrenamiento')
plt.plot(test_accuracies, label='Prueba')
plt.title("Precisión por Época")
plt.xlabel("Épocas")
plt.ylabel("Precisión (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Parte 3: Implementación con scikit-learn (MLPClassifier) 

# Red neuronal equivalente: 1 capa oculta de 16 neuronas, activación ReLU, SGD
clf = MLPClassifier(
    hidden_layer_sizes=(16,),        
    activation='relu',            
    solver='sgd',                   
    learning_rate_init=0.01,        
    max_iter=100,                    
    random_state=42,
    verbose=True                    
)

# Entrenar el modelo
clf.fit(X_train_scaled, y_train)

# Evaluación
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

acc_train_sklearn = accuracy_score(y_train, y_pred_train) * 100
acc_test_sklearn = accuracy_score(y_test, y_pred_test) * 100

print(f"Precisión en entrenamiento (scikit-learn): {acc_train_sklearn:.2f}%")
print(f"Precisión en prueba (scikit-learn): {acc_test_sklearn:.2f}%")

# Matriz de confusión
cm_sklearn = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión - scikit-learn")
plt.grid(False)
plt.show()

