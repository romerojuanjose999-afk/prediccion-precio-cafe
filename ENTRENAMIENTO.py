# ENTRENAMIENTO.py  |  Paso 5 de 6
# Entrenar la neurona de clasificación binaria


import numpy as np
import pickle
from PROCESAMIENTO import procesar_datos

# Funciones de la neurona 

def sigmoide(z):
    # Convierte cualquier número a un valor entre 0 y 1
    return 1 / (1 + np.exp(-z))

def calcular_error(pred, real):
    # Mide qué tan equivocada está la neurona (binary cross-entropy)
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    return -np.mean(real * np.log(pred) + (1 - real) * np.log(1 - pred))

def barra_progreso(ep, total, error, ancho=30):
    lleno = int(ancho * ep / total)
    barra = "█" * lleno + "░" * (ancho - lleno)
    pct   = ep / total * 100
    return f"  [{barra}] {pct:5.1f}%  error: {error:.4f}"

#  Entrenamiento 

def entrenar():

    print("=" * 50)
    print("  PASO 5 — ENTRENAMIENTO DE LA NEURONA")
    print("=" * 50)

    X_train, X_test, Y_train, Y_test = procesar_datos()
    Y_train = Y_train.reshape(-1, 1)
    Y_test  = Y_test.reshape(-1, 1)

    # Inicializar pesos y sesgo con valores pequeños aleatorios
    np.random.seed(42)
    pesos  = np.random.randn(X_train.shape[1], 1) * 0.01
    sesgo  = 0.0
    lr     = 0.01    # tasa de aprendizaje
    epocas = 300

    print(f"\n  Configuración:")
    print(f"    Features de entrada : {X_train.shape[1]}")
    print(f"    Días entrenamiento  : {len(X_train)}")
    print(f"    Épocas              : {epocas}")
    print(f"    Tasa de aprendizaje : {lr}")
    print(f"\n  Entrenando...\n")

    errores_train = []
    errores_test  = []

    for epoca in range(epocas):

        # FORWARD PASS: la neurona predice 
        z    = X_train @ pesos + sesgo
        pred = sigmoide(z)

        # CALCULAR ERROR 
        error_train = calcular_error(pred, Y_train)
        pred_test   = sigmoide(X_test @ pesos + sesgo)
        error_test  = calcular_error(pred_test, Y_test)

        errores_train.append(float(error_train))
        errores_test.append(float(error_test))

        # BACKPROPAGATION: ajustar pesos
        diferencia = pred - Y_train
        pesos -= lr * (X_train.T @ diferencia) / len(X_train)
        sesgo -= lr * diferencia.mean()

        # Mostrar progreso cada 30 épocas
        if (epoca + 1) % 30 == 0 or epoca == 0:
            print(barra_progreso(epoca + 1, epocas, error_train))

    # EVALUACIÓN FINAL
    pred_final = sigmoide(X_test @ pesos + sesgo)
    pred_clase = (pred_final >= 0.5).astype(int)
    precision  = (pred_clase == Y_test).mean() * 100

    VP = int(((pred_clase == 1) & (Y_test == 1)).sum())
    VN = int(((pred_clase == 0) & (Y_test == 0)).sum())
    FP = int(((pred_clase == 1) & (Y_test == 0)).sum())
    FN = int(((pred_clase == 0) & (Y_test == 1)).sum())

    # Importancia de variables
    w_abs  = np.abs(pesos.flatten())
    w_norm = (w_abs / w_abs.sum() * 100).tolist()

    print(f"\n  {'=' * 46}")
    print(f"  RESULTADO FINAL")
    print(f"  {'=' * 46}")
    print(f"  Precisión          : {precision:.1f}%")
    print(f"  Error final        : {errores_train[-1]:.4f}")
    print(f"  VP (sube correcto) : {VP}")
    print(f"  VN (baja correcto) : {VN}")
    print(f"  FP (sube incorrecto): {FP}")
    print(f"  FN (baja incorrecto): {FN}")
    print(f"  {'=' * 46}\n")

    # GUARDAR MODELO 
    modelo = {
        "pesos":         pesos,
        "sesgo":         sesgo,
        "precision":     precision,
        "VP": VP, "VN": VN, "FP": FP, "FN": FN,
        "errores_train": errores_train,
        "errores_test":  errores_test,
        "w_norm":        w_norm,
    }
    with open("modelo_cafe.pkl", "wb") as f:
        pickle.dump(modelo, f)

    print(f"  Guardado: modelo_cafe.pkl")
    print("  OK\n")

if __name__ == "__main__":
    entrenar()
