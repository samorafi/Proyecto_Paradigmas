import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

warnings.filterwarnings('ignore')


# Preprocesa los datos: rellena nulos y escala las columnas numéricas
def preprocesar(df):
    numericas = df.select_dtypes(include='number')

    if numericas.empty:
        return None, "No hay columnas numéricas para hacer clustering."

    if len(numericas) < 5:
        return None, "Se necesitan al menos 5 filas para hacer clustering."

    # rellenamos nulos con la media de cada columna
    numericas = numericas.fillna(numericas.mean())

    # eliminamos columnas que quedaron todas iguales (varianza cero)
    numericas = numericas.loc[:, numericas.std() > 0]

    if numericas.empty:
        return None, "Todas las columnas numéricas son constantes, no se puede hacer clustering."

    # escalamos para que ninguna columna domine por tener valores más grandes
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(numericas)

    return datos_escalados, numericas.columns.tolist()


# Aplica el método del codo para encontrar el k óptimo de K-means
def metodo_codo(datos, max_k=10):
    # el máximo k que tiene sentido es el número de filas dividido 2
    limite = min(max_k + 1, len(datos) // 2 + 1, 11)
    rango_k = list(range(2, limite))

    if len(rango_k) < 2:
        return rango_k, [], 2

    inercias = []
    for k in rango_k:
        # entrenamos y guardamos la inercia (suma de distancias al centroide)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(datos)
        inercias.append(km.inertia_)

    # buscamos el "codo": el k donde la inercia deja de bajar rápido
    mejor_k = rango_k[0]
    if len(inercias) >= 3:
        diffs = np.diff(inercias)
        segunda_derivada = np.diff(diffs)
        # el máximo de la segunda derivada es donde está el codo
        idx_codo = int(np.argmax(segunda_derivada))
        mejor_k = rango_k[idx_codo + 1]

    return rango_k, inercias, mejor_k


# Aplica K-means con el número de clusters especificado
def aplicar_kmeans(datos, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = km.fit_predict(datos)

    # el score de silueta mide qué tan bien separados quedaron los grupos (de -1 a 1)
    if k > 1 and len(datos) > k:
        score = silhouette_score(datos, etiquetas)
    else:
        score = 0.0

    return etiquetas, round(score, 3)


# Aplica DBSCAN como alternativa (no necesita definir k de antemano)
def aplicar_dbscan(datos):
    dbscan = DBSCAN(eps=0.5, min_samples=max(3, len(datos) // 50))
    etiquetas = dbscan.fit_predict(datos)

    # DBSCAN marca como -1 los puntos que no entran en ningún grupo (ruido)
    n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    n_ruido = list(etiquetas).count(-1)

    return etiquetas, n_clusters, n_ruido


# Reduce dimensionalidad a 2D con PCA para poder graficar
def reducir_pca(datos):
    # si ya tiene 2 columnas o menos, no hace falta reducir
    n_componentes = min(2, datos.shape[1])
    pca = PCA(n_components=n_componentes, random_state=42)
    datos_2d = pca.fit_transform(datos)

    # si solo hay 1 componente, agregamos una columna de ceros
    if n_componentes == 1:
        datos_2d = np.column_stack([datos_2d, np.zeros(len(datos_2d))])
        varianza = np.array([pca.explained_variance_ratio_[0], 0.0])
    else:
        varianza = pca.explained_variance_ratio_

    return datos_2d, varianza


# Describe cada cluster mostrando el promedio de sus variables
def describir_clusters(df_original, etiquetas, columnas):
    df_temp = df_original[columnas].copy()
    df_temp = df_temp.fillna(df_temp.mean())

    # agregamos la etiqueta de cluster a cada fila
    df_temp['Cluster'] = etiquetas

    # calculamos el promedio de cada variable por cluster
    descripcion = df_temp.groupby('Cluster').mean().round(2)

    # renombramos los índices para que sea más legible
    nuevos_indices = []
    for i in descripcion.index:
        nuevos_indices.append(f"Grupo {i}" if i != -1 else "Ruido (DBSCAN)")
    descripcion.index = nuevos_indices

    return descripcion


# Genera el gráfico de dispersión de los clusters en 2D
def grafico_clusters(datos_2d, etiquetas, varianza):
    df_plot = pd.DataFrame({
        'PC1': datos_2d[:, 0],
        'PC2': datos_2d[:, 1],
        'Cluster': [f"Grupo {e}" if e != -1 else "Ruido" for e in etiquetas]
    })

    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        title='Visualización de clusters (reducción PCA a 2D)',
        labels={
            'PC1': f'Componente 1 ({varianza[0]*100:.1f}% varianza explicada)',
            'PC2': f'Componente 2 ({varianza[1]*100:.1f}% varianza explicada)'
        },
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=500)

    return fig
