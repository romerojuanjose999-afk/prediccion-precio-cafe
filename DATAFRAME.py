# DATAFRAME.py  |  Paso 2 de 6
# Cargar el CSV y verificar la estructura del DataFrame

import pandas as pd

def convertir_dataframe():

    print("=" * 50)
    print("  PASO 2 — CONVERTIR A DATAFRAME")
    print("=" * 50)

    # Cargar el CSV generado por API_FETCH
    df = pd.read_csv("datos_cafe_crudo.csv", index_col=0, parse_dates=True)

    # Eliminar filas extra que deja yfinance (Price, Ticker)
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]

    # Convertir todas las columnas a números
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\n  Filas    : {df.shape[0]}")
    print(f"  Columnas : {df.shape[1]}")
    print(f"  Nombres  : {list(df.columns)}")

    print(f"\n  Tipos de datos (deben ser float64 / int64):")
    for col, dtype in df.dtypes.items():
        ok = "OK" if "float" in str(dtype) or "int" in str(dtype) else "REVISAR"
        print(f"    {col:<10} {str(dtype):<12} {ok}")

    print(f"\n  Primeras 3 filas:")
    print(df.head(3).to_string())
    print(f"\n  Últimas 3 filas:")
    print(df.tail(3).to_string())

    df.to_csv("datos_cafe_dataframe.csv")
    print(f"\n  Guardado en: datos_cafe_dataframe.csv")
    print("  OK\n")

    return df

if __name__ == "__main__":
    convertir_dataframe()
