import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Genera histogramas para cada columna numérica del dataset
def histogramas(df):
    numericas = df.select_dtypes(include='number')

    if numericas.empty:
        return None

    cols = numericas.columns.tolist()
    n = len(cols)

    # distribuimos en hasta 3 columnas por fila
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols)

    for i, col in enumerate(cols):
        fila = i // n_cols + 1
        columna = i % n_cols + 1

        # agregamos el histograma de cada variable
        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                name=col,
                showlegend=False,
                marker_color='steelblue',
                opacity=0.8
            ),
            row=fila, col=columna
        )

    fig.update_layout(
        title_text='Distribución de variables numéricas',
        height=320 * n_rows,
        bargap=0.1
    )

    return fig


# Genera gráficos de barras para cada columna categórica
def barras(df, tipos=None):
    # determinamos qué columnas mostrar
    if tipos:
        cats = [col for col, tipo in tipos.items() if tipo == 'categórica']
    else:
        cats = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cats:
        return None

    n = len(cats)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cats)

    for i, col in enumerate(cats):
        fila = i // n_cols + 1
        columna = i % n_cols + 1

        # mostramos solo los 15 valores más frecuentes para no saturar
        conteo = df[col].value_counts().head(15)

        fig.add_trace(
            go.Bar(
                x=conteo.index.astype(str),
                y=conteo.values,
                name=col,
                showlegend=False,
                marker_color='coral',
                opacity=0.85
            ),
            row=fila, col=columna
        )

    fig.update_layout(
        title_text='Distribución de variables categóricas',
        height=420 * n_rows
    )

    return fig


# Genera boxplots para ver distribución y outliers de las columnas numéricas
def boxplots(df):
    numericas = df.select_dtypes(include='number')

    if numericas.empty:
        return None

    # "derretimos" el dataframe para tener todas las columnas en un solo gráfico
    df_melted = numericas.melt(var_name='Columna', value_name='Valor')

    fig = px.box(
        df_melted,
        x='Columna',
        y='Valor',
        color='Columna',
        title='Boxplots por columna numérica (los puntos de afuera son posibles outliers)',
        points='outliers'
    )

    fig.update_layout(showlegend=False, height=500)

    return fig


# Genera la scatter matrix (todas las columnas numéricas cruzadas entre sí)
def scatter_matrix(df):
    numericas = df.select_dtypes(include='number')

    if numericas.shape[1] < 2:
        return None

    # limitamos a 6 columnas máximo para que el gráfico no quede ilegible
    cols = numericas.columns.tolist()[:6]

    fig = px.scatter_matrix(
        df[cols],
        title='Scatter matrix — relaciones entre variables numéricas',
        opacity=0.5,
        color_discrete_sequence=['steelblue']
    )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=700)

    return fig
