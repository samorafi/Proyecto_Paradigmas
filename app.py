import gradio as gr
import pandas as pd

from modules import loader, profiler, descriptive, correlations, outliers, clustering, visualizations, report


# ---- funciones de cada pestaña ----

# Carga el archivo y muestra una preview con los tipos detectados
def cargar_datos(files):
    if not files:
        return None, "Por favor subí al menos un archivo.", None, None

    df, mensaje = loader.cargar_multiples(files)

    if df is None:
        return None, mensaje, None, None

    # mostramos las primeras 10 filas para que el usuario confirme que cargó bien
    preview = df.head(10)

    # detectamos los tipos de columnas automáticamente
    tabla_tipos, _ = profiler.resumen_tipos(df)

    return df, mensaje, preview, tabla_tipos


# Calcula estadísticas descriptivas (numéricas y categóricas)
def analizar_estadisticas(df):
    if df is None:
        return "Primero cargá un archivo en la pestaña Datos.", None, None

    try:
        # estadísticas para columnas numéricas
        stats_df, msg = descriptive.calcular_estadisticas(df)

        # resumen para columnas categóricas
        cats_df = descriptive.resumen_categoricas(df)

        return msg, stats_df, cats_df

    except Exception as e:
        return f"Error inesperado: {str(e)}", None, None


# Calcula correlaciones de Pearson y Spearman y genera el heatmap
def analizar_correlaciones(df):
    if df is None:
        return "Primero cargá un archivo.", None, None, None

    try:
        # calculamos Pearson y Spearman
        pearson, msg_p = correlations.calcular_pearson(df)
        spearman, _ = correlations.calcular_spearman(df)

        if pearson is None:
            return msg_p, None, None, None

        # los 5 pares con mayor correlación (en valor absoluto)
        top = correlations.top_correlaciones(pearson)
        top_df = pd.DataFrame(top) if top else pd.DataFrame(
            columns=['Variable 1', 'Variable 2', 'Correlación']
        )

        # heatmap de Pearson
        fig_pearson = correlations.generar_heatmap(pearson)

        # tabla de Spearman para comparar
        spearman_df = spearman.round(3).reset_index() if spearman is not None else None

        return msg_p, top_df, fig_pearson, spearman_df

    except Exception as e:
        return f"Error: {str(e)}", None, None, None


# Detecta outliers con los tres métodos y muestra los resultados
def analizar_outliers(df):
    if df is None:
        return "Primero cargá un archivo.", None, None, None

    try:
        resultados, resumen = outliers.resumen_outliers(df)

        # tabla comparativa de cuántos encontró cada método
        resumen_df = pd.DataFrame([
            {'Método': metodo, 'Outliers encontrados': cantidad,
             '% del total': f"{round(cantidad / len(df) * 100, 1)}%"}
            for metodo, cantidad in resumen.items()
        ])

        # tablas detalladas por método
        zscore_df = resultados['zscore'] if not resultados['zscore'].empty else None
        iqr_df = resultados['iqr'] if not resultados['iqr'].empty else None

        return "Análisis de outliers completado.", resumen_df, zscore_df, iqr_df

    except Exception as e:
        return f"Error: {str(e)}", None, None, None


# Aplica K-means y DBSCAN y muestra los clusters en 2D
def analizar_clustering(df):
    if df is None:
        return "Primero cargá un archivo.", None, None, None

    try:
        # preprocesamos: llenamos nulos y escalamos
        datos, columnas = clustering.preprocesar(df)

        if datos is None:
            # columnas acá es el mensaje de error
            return columnas, None, None, None

        # buscamos el k óptimo con el método del codo
        rango_k, inercias, mejor_k = clustering.metodo_codo(datos)

        # aplicamos K-means con el mejor k
        etiquetas, score = clustering.aplicar_kmeans(datos, mejor_k)

        # reducimos a 2D con PCA para graficar
        datos_2d, varianza = clustering.reducir_pca(datos)

        # generamos el gráfico
        fig = clustering.grafico_clusters(datos_2d, etiquetas, varianza)

        # describimos cada grupo con sus promedios
        descripcion = clustering.describir_clusters(df, etiquetas, columnas)

        msg = (
            f"Se encontraron {mejor_k} grupos "
            f"(score de silueta: {score} — cuanto más cerca de 1, mejor la separación)."
        )
        return msg, descripcion, fig, mejor_k

    except Exception as e:
        return f"Error: {str(e)}", None, None, None


# Genera los gráficos automáticos de distribución
def generar_graficos(df):
    if df is None:
        return "Primero cargá un archivo.", None, None, None, None

    try:
        # detectamos los tipos para pasarlos a la función de barras
        tipos = profiler.detectar_tipos(df)

        fig_hist = visualizations.histogramas(df)
        fig_barras = visualizations.barras(df, tipos)
        fig_box = visualizations.boxplots(df)
        fig_scatter = visualizations.scatter_matrix(df)

        return "Gráficos generados.", fig_hist, fig_barras, fig_box, fig_scatter

    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None


# Exporta el reporte completo en HTML
def exportar_html(df, n_clusters):
    if df is None:
        return None, "Primero cargá un archivo."

    try:
        # recalculamos los datos necesarios para el reporte
        stats_df, _ = descriptive.calcular_estadisticas(df)
        pearson, _ = correlations.calcular_pearson(df)
        top_corr = correlations.top_correlaciones(pearson) if pearson is not None else []
        _, resumen_out = outliers.resumen_outliers(df)
        tipos = profiler.detectar_tipos(df)

        # generamos los insights en texto
        insights = report.generar_insights(df, stats_df, top_corr, resumen_out, n_clusters)

        # generamos las figuras para incluir en el reporte
        figuras = {
            'Histogramas': visualizations.histogramas(df),
            'Boxplots': visualizations.boxplots(df),
            'Barras (categóricas)': visualizations.barras(df, tipos),
            'Scatter matrix': visualizations.scatter_matrix(df),
        }

        ruta = report.generar_html(
            df, stats_df, pearson, top_corr, resumen_out, n_clusters, insights, figuras
        )
        return ruta, "Reporte HTML generado. Hacé clic en el archivo para descargarlo."

    except Exception as e:
        return None, f"Error al generar el HTML: {str(e)}"


# Exporta el reporte en PDF
def exportar_pdf(df):
    if df is None:
        return None, "Primero cargá un archivo."

    try:
        stats_df, _ = descriptive.calcular_estadisticas(df)
        pearson, _ = correlations.calcular_pearson(df)
        top_corr = correlations.top_correlaciones(pearson) if pearson is not None else []
        _, resumen_out = outliers.resumen_outliers(df)

        insights = report.generar_insights(df, stats_df, top_corr, resumen_out, None)

        ruta = report.generar_pdf(df, insights, stats_df)

        if ruta is None:
            return None, "No se pudo generar el PDF. Instalá fpdf2 con: pip install fpdf2"

        return ruta, "Reporte PDF generado."

    except Exception as e:
        return None, f"Error al generar el PDF: {str(e)}"


# ---- interfaz con Gradio ----

_theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    secondary_hue="blue",
).set(
    body_background_fill="#0d0d0d",
    body_background_fill_dark="#0d0d0d",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1e3a5f",
    block_border_color_dark="#1e3a5f",
    input_background_fill="#111827",
    input_background_fill_dark="#111827",
    button_primary_background_fill="#1d4ed8",
    button_primary_background_fill_hover="#1e40af",
    button_primary_background_fill_dark="#1d4ed8",
    button_primary_background_fill_hover_dark="#1e40af",
    button_primary_text_color="#ffffff",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
)

_css = "footer { display: none !important; }"

with gr.Blocks(title="Analizador de Datos con IA", theme=_theme, css=_css) as app:

    # estado compartido: guardamos el dataframe y el número de clusters
    estado_df = gr.State(None)
    estado_n_clusters = gr.State(None)

    gr.Markdown(
        "# Analizador de Datos con IA\n"
        "Subí uno o varios archivos CSV o Excel y la app los analiza automáticamente."
    )

    with gr.Tabs():

        # ---- pestaña 1: cargar datos ----
        with gr.Tab("Datos"):
            gr.Markdown("### Subí tu archivo acá")

            file_input = gr.File(
                label="Arrastrá o seleccioná archivos (CSV o Excel)",
                file_count="multiple",
                file_types=[".csv", ".xlsx", ".xls"]
            )

            btn_cargar = gr.Button("Cargar datos", variant="primary")
            msg_carga = gr.Textbox(label="Estado", interactive=False)

            with gr.Row():
                preview_tabla = gr.Dataframe(
                    label="Vista previa (primeras 10 filas)",
                    interactive=False
                )
                tipos_tabla = gr.Dataframe(
                    label="Tipos de columnas detectados",
                    interactive=False
                )

            btn_cargar.click(
                fn=cargar_datos,
                inputs=[file_input],
                outputs=[estado_df, msg_carga, preview_tabla, tipos_tabla]
            )

        # ---- pestaña 2: estadísticas ----
        with gr.Tab("Estadísticas"):
            gr.Markdown("### Estadísticas descriptivas por columna")

            btn_stats = gr.Button("Calcular estadísticas", variant="primary")
            msg_stats = gr.Textbox(label="Estado", interactive=False)

            stats_tabla = gr.Dataframe(label="Variables numéricas", interactive=False)
            cats_tabla = gr.Dataframe(label="Variables categóricas", interactive=False)

            btn_stats.click(
                fn=analizar_estadisticas,
                inputs=[estado_df],
                outputs=[msg_stats, stats_tabla, cats_tabla]
            )

        # ---- pestaña 3: correlaciones ----
        with gr.Tab("Correlaciones"):
            gr.Markdown("### Correlaciones entre variables numéricas")

            btn_corr = gr.Button("Calcular correlaciones", variant="primary")
            msg_corr = gr.Textbox(label="Estado", interactive=False)

            top_corr_tabla = gr.Dataframe(
                label="Top 5 pares más correlacionados (Pearson)",
                interactive=False
            )

            heatmap_plot = gr.Plot(label="Heatmap de correlación")

            spearman_tabla = gr.Dataframe(
                label="Matriz de correlación de Spearman (más robusta ante outliers)",
                interactive=False
            )

            btn_corr.click(
                fn=analizar_correlaciones,
                inputs=[estado_df],
                outputs=[msg_corr, top_corr_tabla, heatmap_plot, spearman_tabla]
            )

        # ---- pestaña 4: outliers ----
        with gr.Tab("Outliers"):
            gr.Markdown(
                "### Detección de valores atípicos\n"
                "Se usan 3 métodos distintos: Z-score, IQR y un algoritmo de machine learning."
            )

            btn_out = gr.Button("Detectar outliers", variant="primary")
            msg_out = gr.Textbox(label="Estado", interactive=False)

            resumen_out_tabla = gr.Dataframe(
                label="Comparativa entre métodos",
                interactive=False
            )

            with gr.Row():
                zscore_tabla = gr.Dataframe(
                    label="Detalle — Z-score",
                    interactive=False
                )
                iqr_tabla = gr.Dataframe(
                    label="Detalle — IQR",
                    interactive=False
                )

            btn_out.click(
                fn=analizar_outliers,
                inputs=[estado_df],
                outputs=[msg_out, resumen_out_tabla, zscore_tabla, iqr_tabla]
            )

        # ---- pestaña 5: clustering ----
        with gr.Tab("Clusters"):
            gr.Markdown(
                "### Agrupamiento de datos\n"
                "La app busca el número ideal de grupos automáticamente y los grafica en 2D."
            )

            btn_cluster = gr.Button("Aplicar clustering", variant="primary")
            msg_cluster = gr.Textbox(label="Estado", interactive=False)

            desc_cluster = gr.Dataframe(
                label="Descripción de grupos (promedios por cluster)",
                interactive=False
            )

            plot_cluster = gr.Plot(label="Visualización de clusters en 2D")

            btn_cluster.click(
                fn=analizar_clustering,
                inputs=[estado_df],
                outputs=[msg_cluster, desc_cluster, plot_cluster, estado_n_clusters]
            )

        # ---- pestaña 6: gráficos ----
        with gr.Tab("Gráficos"):
            gr.Markdown("### Visualizaciones automáticas (interactivas)")

            btn_graficos = gr.Button("Generar gráficos", variant="primary")
            msg_graficos = gr.Textbox(label="Estado", interactive=False)

            plot_hist = gr.Plot(label="Histogramas (variables numéricas)")
            plot_barras = gr.Plot(label="Barras (variables categóricas)")
            plot_box = gr.Plot(label="Boxplots")
            plot_scatter = gr.Plot(label="Scatter matrix")

            btn_graficos.click(
                fn=generar_graficos,
                inputs=[estado_df],
                outputs=[msg_graficos, plot_hist, plot_barras, plot_box, plot_scatter]
            )

        # ---- pestaña 7: reporte ----
        with gr.Tab("Reporte"):
            gr.Markdown(
                "### Exportar reporte completo\n"
                "El HTML incluye todos los gráficos interactivos. "
                "El PDF es una versión simplificada sin gráficos."
            )

            with gr.Row():
                btn_html = gr.Button("Generar reporte HTML", variant="primary")
                btn_pdf = gr.Button("Generar reporte PDF", variant="secondary")

            msg_reporte = gr.Textbox(label="Estado", interactive=False)

            with gr.Row():
                archivo_html = gr.File(label="Reporte HTML")
                archivo_pdf = gr.File(label="Reporte PDF")

            btn_html.click(
                fn=exportar_html,
                inputs=[estado_df, estado_n_clusters],
                outputs=[archivo_html, msg_reporte]
            )

            btn_pdf.click(
                fn=exportar_pdf,
                inputs=[estado_df],
                outputs=[archivo_pdf, msg_reporte]
            )


if __name__ == "__main__":
    app.launch()
