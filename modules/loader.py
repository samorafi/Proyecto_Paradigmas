import pandas as pd
import os


# Lee un archivo CSV o Excel y devuelve un DataFrame de pandas
def cargar_archivo(file):
    try:
        # obtenemos la ruta del archivo (Gradio puede pasar string o objeto con .name)
        ruta = file.name if hasattr(file, 'name') else str(file)
        extension = os.path.splitext(ruta)[1].lower()

        # si la extensión no se reconoce en el path temporal, intentamos con CSV
        if extension not in ['.csv', '.xlsx', '.xls']:
            extension = '.csv'

        # leemos según el tipo de archivo
        if extension == '.csv':
            # intentamos UTF-8 primero, si falla probamos latin-1
            try:
                df = pd.read_csv(ruta, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(ruta, encoding='latin-1')

        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(ruta)

        else:
            return None, f"Formato no soportado: {extension}. Usá CSV o Excel (.xlsx, .xls)."

        # revisamos que el archivo no esté vacío
        if df.empty:
            return None, "El archivo está vacío, no hay datos para analizar."

        # muy pocas filas no tienen sentido para analizar
        if len(df) < 2:
            return None, "El archivo tiene muy pocas filas para analizar (mínimo 2)."

        return df, f"✅ Archivo cargado: {len(df)} filas × {len(df.columns)} columnas"

    except Exception as e:
        return None, f"❌ Error al leer el archivo: {str(e)}"


# Carga varios archivos y los combina si tienen las mismas columnas
def cargar_multiples(files):
    if not files:
        return None, "No se subió ningún archivo."

    # si es solo uno lo procesamos directo
    if len(files) == 1:
        return cargar_archivo(files[0])

    # cargamos cada archivo por separado y guardamos los que funcionan
    dataframes = []
    errores = []

    for f in files:
        df, msg = cargar_archivo(f)
        if df is not None:
            dataframes.append(df)
        else:
            errores.append(msg)

    if not dataframes:
        return None, "No se pudo cargar ningún archivo.\n" + "\n".join(errores)

    # intentamos combinar todos en uno
    try:
        df_combinado = pd.concat(dataframes, ignore_index=True)
        msg = f"✅ Se combinaron {len(dataframes)} archivos: {len(df_combinado)} filas × {len(df_combinado.columns)} columnas"
        return df_combinado, msg
    except Exception:
        # si las columnas son distintas, usamos solo el primero y avisamos
        return dataframes[0], "⚠️ Los archivos tienen columnas distintas, se usó solo el primero."
