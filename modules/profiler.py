import pandas as pd


# Detecta automáticamente el tipo de dato de cada columna
def detectar_tipos(df):
    tipos = {}

    for col in df.columns:
        # sacamos los nulos para analizar mejor los valores reales
        serie = df[col].dropna()

        if len(serie) == 0:
            tipos[col] = 'vacía'
            continue

        # revisamos si es booleana (solo True/False o 0/1)
        if df[col].dtype == bool:
            tipos[col] = 'booleana'
            continue

        valores_unicos = set(serie.unique())
        if valores_unicos.issubset({True, False, 0, 1, '0', '1', 'True', 'False', 'true', 'false'}):
            if len(valores_unicos) <= 2:
                tipos[col] = 'booleana'
                continue

        # revisamos si ya es numérica por dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            tipos[col] = 'numérica'
            continue

        # revisamos si ya es datetime por dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            tipos[col] = 'fecha'
            continue

        # si es texto, intentamos convertirla a fecha
        if df[col].dtype == object:
            try:
                # usamos solo las primeras 20 filas para no tardar mucho
                pd.to_datetime(serie.head(20), infer_datetime_format=True)
                tipos[col] = 'fecha'
                continue
            except Exception:
                pass

        # si llegamos acá, es categórica (texto que no es fecha)
        tipos[col] = 'categórica'

    return tipos


# Genera una tabla resumen con el tipo y estadísticas básicas de cada columna
def resumen_tipos(df):
    tipos = detectar_tipos(df)

    filas = []
    for col in df.columns:
        # contamos cuántos valores nulos hay
        nulos = df[col].isnull().sum()
        porc_nulos = round(nulos / len(df) * 100, 1)

        # contamos cuántos valores únicos tiene
        unicos = df[col].nunique()

        filas.append({
            'Columna': col,
            'Tipo detectado': tipos[col],
            'Valores únicos': unicos,
            'Nulos': nulos,
            '% Nulos': f"{porc_nulos}%"
        })

    return pd.DataFrame(filas), tipos
