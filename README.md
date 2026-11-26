# ☕ Neurona del Café — ¿Sube o baja el precio mañana?

Red neuronal de clasificación binaria, **programada desde cero con NumPy**, que predice si el
precio del café arábica (contrato **KC=F**, Bolsa de Nueva York) va a **subir o bajar**.
Nació de una necesidad real: como productor de café en el Tolima, saber cuándo conviene vender.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

## Pipeline

```
Yahoo Finance (KC=F, 2015 → hoy)
   │
   ▼ 1. API_FETCH.py            descarga precios diarios        → datos_cafe_crudo.csv
   ▼ 2. DATAFRAME.py            carga y valida la estructura    → datos_cafe_dataframe.csv
   ▼ 3. LIMPIEZA_ESTADISTICA.py limpia y calcula estadísticas   → datos_cafe_limpio.csv
   ▼ 4. PROCESAMIENTO.py        crea features y escala          → datos_cafe_procesado.csv + escalador
   ▼ 5. ENTRENAMIENTO.py        entrena la neurona              → modelo_cafe.pkl
   ▼ 6. EVALUACION.py           predice si sube o baja + matriz de confusión (VP, VN, FP, FN)
```

`DASHBOARD.html` muestra los resultados de forma visual (ábrelo en el navegador).

## Cómo ejecutar

```bash
git clone https://github.com/romerojuanjose999-afk/prediccion-precio-cafe.git
cd prediccion-precio-cafe
pip install -r requirements.txt

python API_FETCH.py
python DATAFRAME.py
python LIMPIEZA_ESTADISTICA.py
python PROCESAMIENTO.py
python ENTRENAMIENTO.py
python EVALUACION.py
```

> Los CSV y modelos `.pkl` no se suben al repositorio: se generan al ejecutar los pasos.

## Aviso

Proyecto académico y de apoyo a decisiones. No es asesoría financiera.

## Autor

**Juan José Romero Hernández** — Ingeniería de Sistemas, UNIMINUTO Ibagué · Productor de café, San Antonio (Tolima).
