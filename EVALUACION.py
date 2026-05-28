# EVALUACION.py  |  Paso 6 de 6
# Predice si el café sube o baja y muestra el resultado

import yfinance as yf
import pandas as pd
import numpy as np
import pickle

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def barra_h(valor, maximo, ancho=20):
    lleno = int(ancho * valor / maximo) if maximo > 0 else 0
    return "█" * lleno + "░" * (ancho - lleno)

def evaluar():

    print("=" * 50)
    print("  PASO 6 — EVALUACIÓN Y PREDICCIÓN")
    print("=" * 50)

    # ── Cargar modelo entrenado ───────────────────────────────
    with open("modelo_cafe.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("escalador_cafe.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("columnas_modelo.pkl", "rb") as f:
        columnas = pickle.load(f)

    pesos     = modelo["pesos"]
    sesgo     = modelo["sesgo"]
    precision = modelo["precision"]
    VP        = modelo["VP"]
    VN        = modelo["VN"]
    FP        = modelo["FP"]
    FN        = modelo["FN"]
    w_norm    = modelo["w_norm"]

    # ── Descargar precio actual ───────────────────────────────
    print(f"\n  Descargando precio actual del café...")
    cafe = yf.download("KC=F", period="60d")
    if isinstance(cafe.columns, pd.MultiIndex):
        cafe.columns = cafe.columns.get_level_values(0)

    df = cafe[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for lag in [1, 2, 3, 7]:
        df[f"precio_lag{lag}"] = df["Close"].shift(lag)
    df["media_7d"]      = df["Close"].rolling(7).mean()
    df["media_14d"]     = df["Close"].rolling(14).mean()
    df["variacion_pct"] = df["Close"].pct_change() * 100
    df["rango_dia"]     = df["High"] - df["Low"]
    df["mes"]           = df.index.month
    df["dia_semana"]    = df.index.dayofweek
    df["dia_del_año"]   = df.index.dayofyear
    df.dropna(inplace=True)

    ultimo    = df[columnas].values[-1].reshape(1, -1)
    ultimo_sc = scaler.transform(ultimo)
    resultado = float(sigmoide(ultimo_sc @ pesos + sesgo)[0][0])
    sube      = resultado >= 0.5
    fecha     = df.index[-1].date()
    precio    = float(df["Close"].values[-1])

    # ── MOSTRAR PREDICCIÓN ────────────────────────────────────
    print(f"\n  {'=' * 46}")
    print(f"  PREDICCIÓN DE HOY")
    print(f"  {'=' * 46}")
    print(f"  Fecha   : {fecha}")
    print(f"  Precio  : {precio:.2f} USCent/lb")
    print()
    if sube:
        print(f"  >>> EL CAFÉ VA A SUBIR  ↑ <<<")
    else:
        print(f"  >>> EL CAFÉ VA A BAJAR  ↓ <<<")
    print()

    # Barra de probabilidad
    prob_pct = resultado * 100
    barra_s  = barra_h(prob_pct, 100, 25)
    barra_b  = barra_h(100 - prob_pct, 100, 25)
    print(f"  SUBE [{barra_s}] {prob_pct:.1f}%")
    print(f"  BAJA [{barra_b}] {100 - prob_pct:.1f}%")
    print(f"  {'=' * 46}\n")

    # ── MÉTRICAS DEL MODELO ───────────────────────────────────
    print(f"  Métricas del modelo:")
    print(f"    Precisión general  : {precision:.1f}%")
    total = VP + VN + FP + FN
    print(f"    VP (sube correcto) : {VP}  ({VP/total*100:.1f}%)")
    print(f"    VN (baja correcto) : {VN}  ({VN/total*100:.1f}%)")
    print(f"    FP (sube error)    : {FP}  ({FP/total*100:.1f}%)")
    print(f"    FN (baja error)    : {FN}  ({FN/total*100:.1f}%)")

    # ── IMPORTANCIA DE VARIABLES ──────────────────────────────
    print(f"\n  Importancia de variables (pesos de la neurona):")
    nombres = [
        "precio ayer", "precio -2d", "precio -3d", "precio -7d",
        "media 7d", "media 14d", "variacion %", "rango dia",
        "mes", "dia semana", "dia del año"
    ]
    pares = sorted(zip(w_norm, nombres), reverse=True)
    for w, n in pares:
        barra = barra_h(w, max(w_norm), 20)
        print(f"    {n:<15} [{barra}] {w:.1f}%")

    # ── ÚLTIMOS 20 DÍAS ───────────────────────────────────────
    print(f"\n  Últimos 10 días de precios:")
    print(f"    {'Fecha':<14} {'Close':>8}  {'Cambio':>8}")
    print(f"    {'─'*35}")
    precios_20 = df["Close"].values[-10:]
    fechas_20  = df.index[-10:]
    for i, (f2, p) in enumerate(zip(fechas_20, precios_20)):
        if i == 0:
            cambio = "—"
        else:
            c = (p - precios_20[i-1]) / precios_20[i-1] * 100
            cambio = f"{'+' if c > 0 else ''}{c:.2f}%"
        print(f"    {str(f2.date()):<14} {p:>8.2f}  {cambio:>8}")

    print(f"\n  OK\n")

if __name__ == "__main__":
    evaluar()
