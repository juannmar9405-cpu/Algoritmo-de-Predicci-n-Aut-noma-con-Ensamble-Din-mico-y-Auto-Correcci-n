import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- 0. Configuración y Datos Ficticios ---
# Simulamos un problema de regresión

print("Cargando y preparando datos ficticios...")
N_SAMPLES = 1000
N_FEATURES = 20
K_SPLITS = 5 # Número de folds para el stacking

# Generamos datos aleatorios para la simulación
# (En un caso real, X serían tus características y 'y' tu objetivo)
X = np.random.rand(N_SAMPLES, N_FEATURES)
# Creamos una 'y' un poco más compleja para que los modelos aprendan algo
y_base = np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * 3) + X[:, 2]**2
y = y_base + (np.random.rand(N_SAMPLES) * 0.1) # Añadimos ruido
y = y.reshape(-1, 1)


# Separamos en entrenamiento (full) y test (hold-out)
# (Usamos 800 para entrenar todo, 200 para prueba final)
X_train_full, y_train_full = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Es crucial escalar los datos para las redes neuronales
scaler_X = StandardScaler()
X_train_full = scaler_X.fit_transform(X_train_full)
X_test = scaler_X.transform(X_test)

# Escalamos 'y' también, lo cual es común en regresión
scaler_y = StandardScaler()
y_train_full = scaler_y.fit_transform(y_train_full)
y_test = scaler_y.transform(y_test)

print(f"Datos listos: {X_train_full.shape} para entrenar, {X_test.shape} para probar.")


# --- 1. Definición de las Arquitecturas de las Redes ---
# (Todas se compilan con Adam y Mean Squared Error para regresión)

def crear_red_n2(input_shape):
    """ Red Base N2: Arquitectura A (ej. MLP simple) """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1) # Salida de regresión
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def crear_red_n3(input_shape):
    """ Red Base N3: Arquitectura B (ej. MLP más profunda) """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Salida de regresión
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def crear_red_n1(input_shape):
    """ Red N1 (Meta-Modelo): Recibe predicciones de N2 y N3 """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)), # Serán 2 entradas (P2, P3)
        layers.Dense(16, activation='relu'),
        layers.Dense(1) # Salida de regresión
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def crear_red_n4(input_shape):
    """ Red N4 (Correctora): Recibe X, P1, P2, P3 """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)), # Serán N_FEATURES + 3 entradas
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Predice el error residual
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 2. Fase 1 y 2: Entrenamiento de N2 y N3 (Stacking K-Fold) ---

print("\n--- Iniciando Fases 1 y 2: Entrenamiento de Modelos Base (N2, N3) con K-Fold ---")

kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=42)

# Arrays para guardar las predicciones "out-of-fold" (OOF)
# Estas se convertirán en los datos de entrenamiento para N1
P2_oof = np.zeros(y_train_full.shape)
P3_oof = np.zeros(y_train_full.shape)

# Listas para guardar los modelos finales para inferencia
modelos_n2_final = []
modelos_n3_final = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    print(f"  Entrenando Fold {fold_idx + 1}/{K_SPLITS}...")
    
    # Partición de datos para este fold
    X_train_fold, y_train_fold = X_train_full[train_idx], y_train_full[train_idx]
    X_val_fold, y_val_fold = X_train_full[val_idx], y_train_full[val_idx]
    
    # Entrenar N2 en este fold
    model_n2_fold = crear_red_n2(N_FEATURES)
    model_n2_fold.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0)
    
    # Entrenar N3 en este fold
    model_n3_fold = crear_red_n3(N_FEATURES)
    model_n3_fold.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0)
    
    # Generar predicciones OOF (out-of-fold)
    P2_oof[val_idx] = model_n2_fold.predict(X_val_fold)
    P3_oof[val_idx] = model_n3_fold.predict(X_val_fold)

print("  Entrenamiento K-Fold completado. Generando predicciones OOF.")

# (Conceptual) Re-entrenar N2 y N3 en *todo* el set de entrenamiento
# Estos son los modelos que se usarán para la inferencia (Paso 5)
print("  Entrenando modelos finales N2 y N3 en todos los datos de entrenamiento...")
modelo_n2_final = crear_red_n2(N_FEATURES)
modelo_n2_final.fit(X_train_full, y_train_full, epochs=25, batch_size=32, verbose=0)

modelo_n3_final = crear_red_n3(N_FEATURES)
modelo_n3_final.fit(X_train_full, y_train_full, epochs=25, batch_size=32, verbose=0)

print("Modelos base finales listos.")

# --- 3. Fase 3: Entrenamiento de N1 (Meta-Modelo) ---

print("\n--- Iniciando Fase 3: Entrenamiento de Meta-Modelo (N1) ---")

# Los datos de entrenamiento para N1 son las predicciones OOF
X_meta_train = np.hstack((P2_oof, P3_oof))
y_meta_train = y_train_full # El objetivo sigue siendo el mismo

modelo_n1 = crear_red_n1(X_meta_train.shape[1])
modelo_n1.fit(X_meta_train, y_meta_train, epochs=20, batch_size=32, verbose=0)

print("Meta-Modelo N1 entrenado.")

# --- 4. Fase 4: Entrenamiento de N4 (Corrector de Error) ---

print("\n--- Iniciando Fase 4: Entrenamiento de Corrector de Error (N4) ---")

# 1. Obtener la predicción de N1 (P1) sobre los datos de entrenamiento
P1_train_oof = modelo_n1.predict(X_meta_train)

# 2. Calcular el error (objetivo de N4)
# y_error_train = y_real - P1_predicha
y_error_train = y_train_full - P1_train_oof

# 3. Crear el set de entrenamiento para N4 (X_original + P1, P2, P3)
X_n4_train = np.hstack((X_train_full, P1_train_oof, P2_oof, P3_oof))

# 4. Entrenar N4
modelo_n4 = crear_red_n4(X_n4_train.shape[1])
modelo_n4.fit(X_n4_train, y_error_train, epochs=25, batch_size=32, verbose=0)

print("Corrector de Error N4 entrenado.")

# --- 5. Fase 5: Evaluación Final (Inferencia en X_test) ---

print("\n--- Iniciando Fase 5: Evaluación en el conjunto de Test (Hold-Out) ---")

# PASO 1: Predicciones de las redes base (N2, N3)
# Usamos los modelos finales entrenados en *todo* X_train_full
P2_test = modelo_n2_final.predict(X_test)
P3_test = modelo_n3_final.predict(X_test)

# PASO 2: Predicción de la red de Stacking (N1)
# N1 usa las predicciones P2_test y P3_test como entrada
X_meta_test = np.hstack((P2_test, P3_test))
P1_test = modelo_n1.predict(X_meta_test)

# PASO 3: Predicción de la red correctora (N4)
# N4 usa X_test + P1, P2, P3 como entrada
X_n4_test = np.hstack((X_test, P1_test, P2_test, P3_test))
P_correccion_test = modelo_n4.predict(X_n4_test)

# PASO 4: Cálculo de la Predicción Final
# Prediccion_Final = Prediccion_N1 + Correccion_N4
y_pred_final = P1_test + P_correccion_test

print("Proceso de inferencia de 4 pasos completado.")

# --- 6. Resultados de la Evaluación ---

print("\n--- Resultados de Evaluación (MSE en Set de Test) ---")
# (Valores más bajos son mejores)

# Volvemos a escalar los datos a su magnitud original para interpretar el error
y_test_orig = scaler_y.inverse_transform(y_test)
P2_test_orig = scaler_y.inverse_transform(P2_test)
P3_test_orig = scaler_y.inverse_transform(P3_test)
P1_test_orig = scaler_y.inverse_transform(P1_test)
y_pred_final_orig = scaler_y.inverse_transform(y_pred_final)

mse_n2 = mean_squared_error(y_test_orig, P2_test_orig)
mse_n3 = mean_squared_error(y_test_orig, P3_test_orig)
mse_n1_stacking = mean_squared_error(y_test_orig, P1_test_orig)
mse_final = mean_squared_error(y_test_orig, y_pred_final_orig)

print(f"  MSE - Solo Red N2 (Arquitectura A): {mse_n2:.6f}")
print(f"  MSE - Solo Red N3 (Arquitectura B): {mse_n3:.6f}")
print(f"  MSE - Red N1 (Solo Stacking):        {mse_n1_stacking:.6f}")
print(f"  MSE - MODELO FINAL (N1 + N4):       {mse_final:.6f}")

print("\n(Idealmente, el MSE del Modelo Final debería ser el más bajo)")

# --- 7. Fase 6: Retroalimentación (Conceptual) ---
# En un sistema real (MLOps):
# 1. Nuevos datos (X_new, y_new) llegan.
# 2. Se generan predicciones (y_pred_final).
# 3. Se calcula el error real: error_real = y_new - y_pred_final
# 4. Estos nuevos pares (X_n4_train, y_error_train) se usan para
#    re-entrenar (fine-tuning) el modelo N4.
# 5. Si los errores son consistentemente altos en ciertos tipos de datos
#    (como N4 predijo), se usan para re-entrenar N2 y N3 (Opción A, B o C).

print("\n--- Simulación Completa ---")
