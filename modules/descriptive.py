import pandas as pd
import numpy as np


# Calcula estadísticas descriptivas para todas las columnas numéricas
def calcular_estadisticas(df):
    # nos quedamos solo con las columnas que tienen números
    numericas = df.select_dtypes(include='number')

    if numericas.empty:
        return None, "No hay columnas numéricas para analizar."

    filas = []

    for col in numericas.columns:
        # sacamos los nulos para calcular sin problemas
        serie = numericas[col].dropna()

        if len(serie) == 0:
            continue

        # calculamos cada estadística una por una
        promedio = round(serie.mean(), 3)
        mediana = round(serie.median(), 3)
        moda = round(serie.mode()[0], 3) if not serie.mode().empty else '-'
        minimo = round(serie.min(), 3)
        maximo = round(serie.max(), 3)
        desvio = round(serie.std(), 3)
        p25 = round(serie.quantile(0.25), 3)
        p75 = round(serie.quantile(0.75), 3)

        # los nulos los calculamos sobre la columna original (no la serie sin nulos)
        nulos = df[col].isnull().sum()
        porc_nulos = f"{round(nulos / len(df) * 100, 1)}%"

        filas.append({
            'Columna': col,
            'Promedio': promedio,
            'Mediana': mediana,
            'Moda': moda,
            'Mín': minimo,
            'Máx': maximo,
            'Desvío std': desvio,
            'P25': p25,
            'P75': p75,
            'Nulos': nulos,
            '% Nulos': porc_nulos
        })

    if not filas:
        return None, "No se pudieron calcular estadísticas."

    return pd.DataFrame(filas), "✅ Estadísticas calculadas correctamente."


# Genera un resumen de las columnas categóricas (texto)
def resumen_categoricas(df):
    # sacamos solo las columnas de texto o categoría
    categoricas = df.select_dtypes(include=['object', 'category'])

    if categoricas.empty:
        return None

    filas = []
    for col in categoricas.columns:
        conteo = df[col].value_counts()

        if conteo.empty:
            continue

        # valor más frecuente y su frecuencia
        valor_top = conteo.idxmax()
        frecuencia = conteo.max()

        filas.append({
            'Columna': col,
            'Valores únicos': df[col].nunique(),
            'Más frecuente': str(valor_top),
            'Frecuencia': frecuencia,
            '% del total': f"{round(frecuencia / len(df) * 100, 1)}%",
            'Nulos': df[col].isnull().sum()
        })

    return pd.DataFrame(filas) if filas else None
