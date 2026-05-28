# API_FETCH.py  |  Paso 1 de 6
# Conectarse a Yahoo Finance y descargar el precio del café

import yfinance as yf
import pandas as pd

def obtener_datos_cafe(fecha_inicio="2015-01-01"):

    print("=" * 50)
    print("  PASO 1 — CONSUMIR LA API")
    print("=" * 50)
    print(f"\n  Fuente  : Yahoo Finance")
    print(f"  Ticker  : KC=F  (Contrato C del café - Bolsa NY)")
    print(f"  Desde   : {fecha_inicio}")
    print(f"  Descargando...\n")

    # Conectarse a Yahoo Finance y descargar datos históricos
    cafe = yf.download("KC=F", start=fecha_inicio)

    # Las versiones nuevas de yfinance devuelven columnas dobles
    # como ("Close", "KC=F") — esto las aplana a solo "Close"
    if isinstance(cafe.columns, pd.MultiIndex):
        cafe.columns = cafe.columns.get_level_values(0)

    # Seleccionar solo las columnas necesarias
    df = cafe[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "date"

    # Convertir a números (por si llegaron como texto)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Guardar en CSV para los siguientes pasos
    df.to_csv("datos_cafe_crudo.csv")

    print(f"  Registros descargados : {len(df)}")
    print(f"  Fecha inicio          : {df.index.min().date()}")
    print(f"  Fecha fin             : {df.index.max().date()}")
    print(f"\n  Tipos de datos:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<10} {dtype}")
    print(f"\n  Últimos 3 registros:")
    print(df.tail(3).to_string())
    print(f"\n  Guardado en: datos_cafe_crudo.csv")
    print("  OK\n")

    return df

if __name__ == "__main__":
    obtener_datos_cafe()
