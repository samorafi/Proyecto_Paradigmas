import pandas as pd
import plotly.io as pio
import tempfile
from datetime import datetime


# Genera frases de insight en español usando los resultados reales del análisis
def generar_insights(df, stats_df, top_corr, resumen_outliers, n_clusters):
    insights = []

    # cuántos datos tiene el dataset
    insights.append(f"El dataset tiene {len(df)} registros y {len(df.columns)} columnas.")

    # estado de los nulos
    total_nulos = df.isnull().sum().sum()
    if total_nulos > 0:
        porc = round(total_nulos / (len(df) * len(df.columns)) * 100, 1)
        insights.append(f"Se encontraron {total_nulos} valores nulos ({porc}% del total de celdas).")
    else:
        insights.append("El dataset no tiene valores nulos.")

    # insight sobre las correlaciones más fuertes
    if top_corr and len(top_corr) > 0:
        mejor = top_corr[0]
        corr_abs = abs(mejor['Correlación'])
        if corr_abs > 0.8:
            fuerza = "muy fuerte"
        elif corr_abs > 0.6:
            fuerza = "fuerte"
        elif corr_abs > 0.4:
            fuerza = "moderada"
        else:
            fuerza = "débil"
        insights.append(
            f"Las variables '{mejor['Variable 1']}' y '{mejor['Variable 2']}' tienen "
            f"una correlación {fuerza} de {mejor['Correlación']}."
        )

    # insight sobre outliers
    if resumen_outliers:
        max_outliers = max(resumen_outliers.values())
        metodo = max(resumen_outliers, key=resumen_outliers.get)
        porc = round(max_outliers / len(df) * 100, 1)
        if max_outliers > 0:
            insights.append(
                f"El método {metodo} detectó {max_outliers} valores atípicos ({porc}% de los registros)."
            )
        else:
            insights.append("No se detectaron valores atípicos significativos en el dataset.")

    # insight sobre clusters
    if n_clusters and n_clusters > 1:
        insights.append(f"Se identificaron {n_clusters} grupos principales en los datos usando K-means.")

    # columna con más variabilidad
    if stats_df is not None and not stats_df.empty and 'Desvío std' in stats_df.columns:
        try:
            idx_max = stats_df['Desvío std'].idxmax()
            col_variable = stats_df.loc[idx_max, 'Columna']
            insights.append(f"La columna '{col_variable}' es la que tiene mayor variabilidad en los datos.")
        except Exception:
            pass

    return insights


# Arma el reporte completo en HTML con todos los gráficos embebidos
def generar_html(df, stats_df, pearson, top_corr, resumen_outliers, n_clusters, insights, figuras):
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")

    # convertimos cada figura de Plotly a HTML (con la librería cargada desde CDN)
    figs_html = ""
    for nombre, fig in figuras.items():
        if fig is not None:
            html_fig = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
            figs_html += f"<h3>{nombre}</h3>\n{html_fig}\n"

    # tabla de estadísticas en HTML
    tabla_stats = ""
    if stats_df is not None and not stats_df.empty:
        tabla_stats = stats_df.to_html(index=False, classes='tabla', border=0)

    # lista de insights como HTML
    lista_insights = "".join([f"<li>{insight}</li>" for insight in insights])

    # armamos el HTML completo
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de análisis de datos</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
            color: #333;
            background: #f9f9f9;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        h3 {{ color: #7f8c8d; }}
        .meta {{ color: #888; margin-bottom: 30px; }}
        .tabla {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .tabla th, .tabla td {{
            border: 1px solid #e0e0e0;
            padding: 10px 12px;
            text-align: left;
        }}
        .tabla th {{ background-color: #3498db; color: white; }}
        .tabla tr:nth-child(even) {{ background-color: #f5f5f5; }}
        .insights {{
            background: #eaf4fb;
            padding: 15px 25px;
            border-left: 4px solid #3498db;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}
        .insights li {{ margin: 8px 0; }}
        .footer {{
            text-align: center;
            color: #bbb;
            margin-top: 50px;
            font-size: 12px;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Reporte de análisis de datos</h1>
    <p class="meta">Generado el {fecha} &nbsp;|&nbsp; {len(df)} registros × {len(df.columns)} columnas</p>

    <h2>Insights principales</h2>
    <div class="insights">
        <ul>{lista_insights}</ul>
    </div>

    <h2>Estadísticas descriptivas</h2>
    {tabla_stats if tabla_stats else "<p>No se calcularon estadísticas.</p>"}

    <h2>Visualizaciones</h2>
    {figs_html if figs_html else "<p>No se generaron gráficos.</p>"}

    <div class="footer">Generado con el Analizador de Datos &mdash; Proyecto estudiantil</div>
</body>
</html>"""

    # guardamos en un archivo temporal y devolvemos la ruta
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.html', delete=False, encoding='utf-8'
    ) as f:
        f.write(html)
        return f.name


# Genera un reporte en PDF usando fpdf2
def generar_pdf(df, insights, stats_df):
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # título principal
        pdf.set_font('Helvetica', 'B', 18)
        pdf.cell(0, 12, 'Reporte de análisis de datos', new_x="LMARGIN", new_y="NEXT")

        # fecha y tamaño
        pdf.set_font('Helvetica', '', 10)
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
        pdf.cell(0, 7, f'Generado el {fecha}', new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f'Dataset: {len(df)} registros x {len(df.columns)} columnas', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

        # sección de insights
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Insights principales', new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('Helvetica', '', 10)

        for insight in insights:
            # multi_cell maneja el salto de línea automáticamente
            pdf.multi_cell(0, 6, f'  - {insight}', new_x="LMARGIN", new_y="NEXT")

        pdf.ln(5)

        # tabla de estadísticas resumida
        if stats_df is not None and not stats_df.empty:
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, 'Estadísticas descriptivas', new_x="LMARGIN", new_y="NEXT")

            # cabecera de la tabla
            columnas_pdf = ['Columna', 'Promedio', 'Mediana', 'Mín', 'Máx', 'Desvío std']
            ancho_col = (pdf.w - pdf.l_margin - pdf.r_margin) / len(columnas_pdf)
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_fill_color(52, 152, 219)
            pdf.set_text_color(255, 255, 255)
            for c in columnas_pdf:
                pdf.cell(ancho_col, 7, str(c), border=1, fill=True)
            pdf.ln()

            # filas de datos
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(0, 0, 0)
            for _, row in stats_df.iterrows():
                for c in columnas_pdf:
                    valor = str(row.get(c, '-'))[:14]
                    pdf.cell(ancho_col, 6, valor, border=1)
                pdf.ln()

        # nota al pie sobre los gráficos
        pdf.ln(8)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(
            0, 6,
            'Nota: Las visualizaciones interactivas están disponibles en el reporte HTML.',
            new_x="LMARGIN", new_y="NEXT"
        )

        # guardamos en archivo temporal
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            nombre = f.name

        pdf.output(nombre)
        return nombre

    except ImportError:
        return None
    except Exception as e:
        raise e
