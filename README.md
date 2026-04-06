## Proyecto

SANCHEZ SEAS KEVIN ANDRES (ksanchez90688@ufide.ac.cr) 
MORA FIGUEROA SEBASTIAN ADOLFO (smora40063@ufide.ac.cr)
PORTUGUEZ ROJAS FIORELLA MARIA (fportuguez30592@ufide.ac.cr) 
TENORIO LOPEZ ISAAC JOSUE (itenorio10388@ufide.ac.cr)

Universidad Fidélitas

Ingeniería en Sistemas de Computación

Paradigmas de Programación SC-250

ROMERO NAVARRO MICHAEL

## Analizador de Datos con IA

Subís un CSV o Excel y la app te dice todo lo que hay adentro: estadísticas, correlaciones, outliers, clusters y gráficos. Sin saber estadística.

## Instalación y uso

**1. Instalá las dependencias**
```bash
pip install -r requirements.txt
```

**2. Corré la app**
```bash
python app.py
```

**3. Abrí el navegador**

La terminal te va a mostrar una URL tipo `http://127.0.0.1:7860` — click y listo.

## ¿Qué hace cada pestaña?

| Pestaña | Qué hace |
| Datos | Subís el archivo y ves una preview |
| Estadísticas | Promedio, mediana, desvío, nulos, etc. |
| Correlaciones | Heatmap y los 5 pares más correlacionados |
| Outliers | Detecta datos raros con 3 métodos distintos |
| Clusters | Agrupa los datos por similitud automáticamente |
| Gráficos | Histogramas, boxplots, scatter matrix |
| Reporte | Descargás todo en HTML o PDF |

## Formatos soportados

- CSV (UTF-8 o Latin-1)
- Excel (.xlsx, .xls)
- Podés subir varios archivos a la vez (los combina si tienen las mismas columnas)