# PROCESAMIENTO.py  |  Paso 4 de 6
# Crear las features y preparar los datos para la neurona   

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def procesar_datos():

    print("=" * 50)
    print("  PASO 4 — PROCESAMIENTO Y FEATURES")
    print("=" * 50)

    df = pd.read_csv("datos_cafe_limpio.csv", index_col=0, parse_dates=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    print(f"\n  Creando features (entradas para la neurona)...")

    # TARGET: lo que la neurona debe aprender a predecir
    # 1 = el precio de mañana sube, 0 = baja
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Features de lag: precios anteriores
    for lag in [1, 2, 3, 7]:
        df[f"precio_lag{lag}"] = df["Close"].shift(lag)
        print(f"    precio_lag{lag}   — precio de hace {lag} día(s)")

    # Media móvil: tendencia reciente
    df["media_7d"]  = df["Close"].rolling(window=7).mean()
    df["media_14d"] = df["Close"].rolling(window=14).mean()
    print(f"    media_7d     — promedio últimos 7 días")
    print(f"    media_14d    — promedio últimos 14 días")

    # Features del día
    df["variacion_pct"] = df["Close"].pct_change() * 100
    df["rango_dia"]     = df["High"] - df["Low"]
    print(f"    variacion_pct — cambio % respecto al día anterior")
    print(f"    rango_dia     — diferencia entre máximo y mínimo")

    # Features temporales
    df["mes"]         = df.index.month
    df["dia_semana"]  = df.index.dayofweek
    df["dia_del_año"] = df.index.dayofyear
    print(f"    mes, dia_semana, dia_del_año — info temporal")

    df.dropna(inplace=True)

    columnas = [
        "precio_lag1", "precio_lag2", "precio_lag3", "precio_lag7",
        "media_7d", "media_14d", "variacion_pct", "rango_dia",
        "mes", "dia_semana", "dia_del_año"
    ]

    X = df[columnas]
    Y = df["target"]

    # Dividir 80% entrenamiento / 20% prueba (sin barajar — orden temporal)
    split   = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    Y_train = Y.iloc[:split]
    Y_test  = Y.iloc[split:]

    print(f"\n  División del dataset:")
    print(f"    Total días      : {len(X)}")
    print(f"    Entrenamiento   : {len(X_train)}  (80%)")
    print(f"    Prueba          : {len(X_test)}   (20%)")

    # Escalar con StandardScaler
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n  Balance del target:")
    print(f"    Días que subió  : {Y.sum()}  ({Y.mean()*100:.1f}%)")
    print(f"    Días que bajó   : {(Y==0).sum()}  ({(Y==0).mean()*100:.1f}%)")

    # Guardar escalador y columnas para usarlos en evaluación
    with open("escalador_cafe.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("columnas_modelo.pkl", "wb") as f:
        pickle.dump(columnas, f)

    df.to_csv("datos_cafe_procesado.csv")
    print(f"\n  Guardado: escalador_cafe.pkl, columnas_modelo.pkl")
    print(f"  Guardado: datos_cafe_procesado.csv")
    print("  OK\n")

    return X_train_sc, X_test_sc, Y_train.values, Y_test.values

if __name__ == "__main__":
    procesar_datos()
