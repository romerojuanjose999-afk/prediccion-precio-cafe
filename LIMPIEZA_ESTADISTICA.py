# LIMPIEZA_ESTADISTICA.py  |  Paso 3 de 6
# Limpiar datos y calcular estadísticas descriptivas

import pandas as pd
import numpy as np

def limpiar_y_estadisticas():

    print("=" * 50)
    print("  PASO 3 — LIMPIEZA Y ESTADÍSTICAS")
    print("=" * 50)

    df = pd.read_csv("datos_cafe_dataframe.csv", index_col=0, parse_dates=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── LIMPIEZA ─────────────────────────────────────────────
    print(f"\n  Filas antes de limpiar : {len(df)}")
    print(f"\n  Valores nulos por columna:")
    for col, n in df.isnull().sum().items():
        print(f"    {col:<10} {n}")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df["Close"] > 0]

    print(f"\n  Filas después de limpiar : {len(df)}")
    print(f"  Valores nulos restantes  : {df.isnull().sum().sum()}")

    # ── ESTADÍSTICAS ─────────────────────────────────────────
    print(f"\n  Estadísticas del precio de cierre (Close):")
    desc = df["Close"].describe()
    for k, v in desc.items():
        print(f"    {k:<10} {v:.2f}")

    print(f"\n  Precio mínimo   : {df['Close'].min():.2f} USCent/lb")
    print(f"  Precio máximo   : {df['Close'].max():.2f} USCent/lb")
    print(f"  Precio promedio : {df['Close'].mean():.2f} USCent/lb")
    print(f"  Desv. estándar  : {df['Close'].std():.2f}")

    # Outliers con método IQR
    Q1  = df["Close"].quantile(0.25)
    Q3  = df["Close"].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    outliers = df[(df["Close"] < lim_inf) | (df["Close"] > lim_sup)]

    print(f"\n  Detección de outliers (método IQR):")
    print(f"    Q1        : {Q1:.2f}")
    print(f"    Q3        : {Q3:.2f}")
    print(f"    IQR       : {IQR:.2f}")
    print(f"    Lím. inf  : {lim_inf:.2f}")
    print(f"    Lím. sup  : {lim_sup:.2f}")
    print(f"    Outliers  : {len(outliers)}")

    df.to_csv("datos_cafe_limpio.csv")
    print(f"\n  Guardado en: datos_cafe_limpio.csv")
    print("  OK\n")

    return df

if __name__ == "__main__":
    limpiar_y_estadisticas()
