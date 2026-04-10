import pandas as pd
import plotly.graph_objects as go


# Calcula la matriz de correlación de Pearson (mide relaciones lineales)
def calcular_pearson(df):
    numericas = df.select_dtypes(include='number')

    if numericas.shape[1] < 2:
        return None, "Se necesitan al menos 2 columnas numéricas para calcular correlaciones."

    # Pearson asume distribución normal y mide relaciones lineales
    matriz = numericas.corr(method='pearson')
    return matriz, "✅ Correlaciones de Pearson calculadas."


# Calcula la correlación de Spearman (más robusta ante outliers)
def calcular_spearman(df):
    numericas = df.select_dtypes(include='number')

    if numericas.shape[1] < 2:
        return None, "Se necesitan al menos 2 columnas numéricas."

    # Spearman usa rangos en vez de valores, por eso no le afectan tanto los outliers
    matriz = numericas.corr(method='spearman')
    return matriz, "✅ Correlaciones de Spearman calculadas."


# Devuelve los N pares de variables con mayor correlación (en valor absoluto)
def top_correlaciones(matriz, n=5):
    if matriz is None or matriz.empty:
        return []

    pares = []
    cols = matriz.columns.tolist()

    # recorremos solo la mitad de la matriz para no duplicar pares
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            valor = matriz.iloc[i, j]

            # ignoramos los NaN que pueden aparecer si una columna es constante
            if pd.isna(valor):
                continue

            pares.append({
                'Variable 1': cols[i],
                'Variable 2': cols[j],
                'Correlación': round(valor, 3)
            })

    if not pares:
        return []

    # ordenamos de mayor a menor por valor absoluto
    pares.sort(key=lambda x: abs(x['Correlación']), reverse=True)

    return pares[:n]


# Genera el heatmap de correlación usando Plotly
def generar_heatmap(matriz, titulo='Mapa de calor de correlaciones (Pearson)'):
    if matriz is None or matriz.empty:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=matriz.values,
        x=matriz.columns.tolist(),
        y=matriz.columns.tolist(),
        # rojo = correlación negativa, azul = correlación positiva
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=matriz.round(2).values,
        texttemplate='%{text}',
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=titulo,
        height=max(400, len(matriz.columns) * 60),
        width=max(500, len(matriz.columns) * 80)
    )

    return fig
