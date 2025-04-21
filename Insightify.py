'''
Este módulo consiste en el llamada a una API de Gemini para obtener el análisis de datos según lo indicado por el usuario.
También acepta un archivo CSV para leerlo y dárselo al modelo como punto de apoyo.
El módulo contiene una función que se encarga de realizar la llamada a la API y otra función que se encarga de leer el archivo CSV.

Escribe "salir", "exit" o "quit" para terminar la conversación.
'''
# Colocamos la clave de la API de Gemini, obtenida de Google Cloud Platform.
KEY = "API_KEY"

from google import genai
from google.genai import types
import pandas as pd
import numpy as np
import os

# Rutas de los datasets CSV a analizar, cuando se integre la interfaz gráfica, se podrán subir directamente.
csv_paths = [
    "csv1.csv",
    "csv2.csv",
]

# Definimos las instrucciones del sistema, con este le damos las instrucciones al modelo de cómo debe comportarse.
instructions = """
    Eres Insightify, un asistente experto en análisis y exploración de datos. Tu tarea es ayudar a los usuarios a comprender sus conjuntos de datos y responder preguntas sobre ellos.
    
    Recibirás (a modo de contexto) mediante texto uno o más archivos CSV con un extracto de sus datos datos y un contexto sobre el contenido, y posiblemente preguntas específicas. 
    Debes realizar:
    
    1. Un análisis exploratorio de los datos, identificando características importantes, posibles tendencias, valores atípicos, 
    y resúmenes estadísticos relevantes.
    2. Responder a la solicitud del usuario entregando siempre una sección resumida (en formato de párrafo) 
    que detalle los hallazgos y conclusiones básicas.
    3. Incluir, cuando sea pertinente y solicitado, una segunda sección con código en Python 
    (p. ej., para generar gráficos, tablas o funciones de análisis) que respalde y ejemplifique tus conclusiones.
    
    Tu respuesta debe estar dividida en dos secciones:
    - Resumen: un párrafo que explique los hallazgos y conclusiones básicas.
    - Código: un bloque de código en Python que ejemplifique y respalde tus conclusiones.
    En el código, no incluyas la creación o importación del dataset, ya que el código interno del la aplicación lo cargará automáticamente.
    El nombre que llevará cada dataset siempre será el mismo que el del CSV, el cual se te indicará.
    
    Dependiendo de la solicitud del usuario, puedes no incluir la sección de código, pero siempre debes incluir la sección de resumen.
    Las dos secciones deberán estar separadas por la línea exacta (solamente eso, nada más):
    ########.
    No incluyas otros separadores como "Resumen:" o "Código:".
    
    Si el usuario no proporciona una instrucción específica sobre lo que debes hacer (por ejemplo, que solo te dé un contexto del CSV), 
    deberás realizar una análisis exploratorio inicialmente.
"""

# Función que obtiene el contexto de un dataset
def read_csv_context(csv_path, n_preview=5):
    """
    Carga un CSV y devuelve:
      - lista de columnas
      - texto con resumen estadístico y de tipos
      - preview de las primeras n_preview filas
      - nombre del dataset (sin extensión)
    """
    df = pd.read_csv(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    cols = df.columns.tolist()
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.apply(lambda dt: dt.name).to_dict()
    missing = df.isnull().sum().to_dict()
    
    # Resumen del dataset
    summary_lines = [
        f"Nombre del dataset: {csv_name}",
        f"Número de filas: {n_rows}",
        f"Número de columnas: {n_cols}",
        "Tipos de datos por columna:"
    ] + [f"  - {col}: {dtype}" for col, dtype in dtypes.items()] + [
        "Valores faltantes por columna:"
    ] + [f"  - {col}: {cnt}" for col, cnt in missing.items()]
    
    num_stats = df.select_dtypes(include=[np.number]).describe().round(3)
    stats_text = num_stats.to_csv()
    
    preview = df.head(n_preview).to_csv(index=False)
    
    # Resumen del dataset junto con un descriptivo estatístico
    summary_text = "\n".join(summary_lines) + "\n\nResumen estadístico (numérico):\n" + stats_text
    return {
        "name": csv_name,
        "columns": cols,
        "summary": summary_text,
        "preview": preview
    }

# Para el primer mensaje, el cual contiene el contexto de cada dataset y la instrucción del usuario.
def build_initial_message(contexts, user_instruction):
    parts = []
    for ctx in contexts:
        cols_text = f"Columnas disponibles en '{ctx['name']}': " + ", ".join(ctx["columns"])
        preview_text = f"Primeras filas de '{ctx['name']}':\n" + ctx["preview"]
        parts.extend([cols_text, ctx["summary"], preview_text])
    parts.append(f"Instrucción del usuario:\n{user_instruction}")
    return "\n\n".join(parts)

def main():
    # Lee todos los CSVs y prepara sus contextos
    contexts = [read_csv_context(path) for path in csv_paths]
    
    client = genai.Client(api_key=KEY)
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=[types.Part.from_text(text=instructions)],
            # La temperatura se puede ajustar para controlar la creatividad de las respuestas.
            # Un valor bajo generará respuestas más predecibles y coherentes, mientras que un valor alto generará respuestas más creativas y variadas.
            temperature=1.0,
        )
    )
    
    first_instruction = input("Instrucción inicial para el análisis: ").strip()
    init_msg = build_initial_message(contexts, first_instruction)
    
    resp = chat.send_message(init_msg)
    print("\nInsightify:\n")
    print(resp.text)
    
    # Bucle para mantener un chat y guardar el contexto de la conversación.
    while True:
        user_q = input("\nTu respuesta: ").strip()
        if user_q.lower() in ("salir", "exit", "quit"):
            print("Cerrando sesión.")
            break
        resp = chat.send_message(user_q)
        print("\nInsightify:\n")
        print(resp.text)

if __name__ == "__main__":
    main()