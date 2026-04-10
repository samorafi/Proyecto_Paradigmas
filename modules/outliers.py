import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest


# Detecta outliers con Z-score: valores muy alejados del promedio
def detectar_zscore(df, umbral=3.0):
    numericas = df.select_dtypes(include='number')

    if numericas.empty:
        return pd.DataFrame()

    resultados = []

    for col in numericas.columns:
        serie = numericas[col].dropna()

        # necesitamos al menos 3 valores para calcular el z-score
        if len(serie) < 3:
            continue

        # calculamos cuántos desvíos estándar está cada valor del promedio
        z_valores = np.abs(stats.zscore(serie))

        # lo convertimos a Series con el mismo índice para indexar bien
        z_series = pd.Series(z_valores, index=serie.index)

        # los que superan el umbral (por defecto 3) son outliers
        for idx in z_series[z_series > umbral].index:
            resultados.append({
                'Índice': idx,
                'Columna': col,
                'Valor': round(df.loc[idx, col], 4),
                'Z-score': round(z_series[idx], 3),
                'Método': 'Z-score'
            })

    return pd.DataFrame(resultados)


# Detecta outliers con IQR: valores fuera del rango intercuartil
def detectar_iqr(df):
    numericas = df.select_dtypes(include='number')

    if numericas.empty:
        return pd.DataFrame()

    resultados = []

    for col in numericas.columns:
        serie = numericas[col].dropna()

        if len(serie) < 4:
            continue

        # calculamos los cuartiles y el rango intercuartil
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        # todo lo que esté fuera de este rango se considera outlier
        limite_bajo = q1 - 1.5 * iqr
        limite_alto = q3 + 1.5 * iqr

        mascara = (serie < limite_bajo) | (serie > limite_alto)

        for idx in serie[mascara].index:
            resultados.append({
                'Índice': idx,
                'Columna': col,
                'Valor': round(df.loc[idx, col], 4),
                'Límite bajo': round(limite_bajo, 3),
                'Límite alto': round(limite_alto, 3),
                'Método': 'IQR'
            })

    return pd.DataFrame(resultados)


# Detecta outliers con Isolation Forest (algoritmo de machine learning)
def detectar_isolation_forest(df, contaminacion=0.05):
    numericas = df.select_dtypes(include='number').dropna()

    if numericas.empty:
        return pd.DataFrame(), "No hay columnas numéricas sin nulos."

    if len(numericas) < 10:
        return pd.DataFrame(), "Se necesitan al menos 10 filas sin nulos para Isolation Forest."

    try:
        # entrenamos el modelo — contamination define qué % esperamos que sean outliers
        modelo = IsolationForest(contamination=contaminacion, random_state=42, n_estimators=100)
        predicciones = modelo.fit_predict(numericas)

        # los que el modelo marca con -1 son los outliers
        mascara_outlier = predicciones == -1
        indices_outlier = numericas.index[mascara_outlier]

        # calculamos el score de anomalía para cada outlier
        scores = modelo.score_samples(numericas)
        scores_series = pd.Series(scores, index=numericas.index)

        resultados = []
        for idx in indices_outlier:
            resultados.append({
                'Índice': idx,
                'Score de anomalía': round(scores_series[idx], 4),
                'Método': 'Isolation Forest'
            })

        msg = f"✅ Isolation Forest encontró {len(resultados)} outliers ({round(len(resultados)/len(df)*100, 1)}%)."
        return pd.DataFrame(resultados), msg

    except Exception as e:
        return pd.DataFrame(), f"❌ Error en Isolation Forest: {str(e)}"


# Junta los tres métodos y devuelve un resumen comparativo
def resumen_outliers(df):
    resultados = {}

    # aplicamos los tres métodos
    resultados['zscore'] = detectar_zscore(df)
    resultados['iqr'] = detectar_iqr(df)
    resultados['isolation_forest'], _ = detectar_isolation_forest(df)

    # contamos cuántos encontró cada uno
    resumen = {
        'Z-score': len(resultados['zscore']),
        'IQR': len(resultados['iqr']),
        'Isolation Forest': len(resultados['isolation_forest'])
    }

    return resultados, resumen
