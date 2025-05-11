import base64
import os
import sys
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from key import KEY

SYSTEM_INSTRUCTIONS = """
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
    - Resumen: un párrafo que explique los hallazgos y conclusiones básicas. No pongas "Resumen:" ni nada similar.
    - Código: un bloque de código en Python que ejemplifique y respalde tus conclusiones. Solo pon el código directamente, sin un indicador como ```python ```.
    Considera que el código que vas a generar será usado de forma temporal, así que no se guardarán los cambios en el dataset ni en otro lado. Considera esto al momento de recibir retroalimentación del usuario.
    Estaremos usando la librería Streamlit para mostrar el código y los resultados, así que asegúrate de que el código sea compatible con ella.
    Por ejemplo, si generas un gráfico, asegúrate de que se muestre correctamente en Streamlit. En lugar de usar plt.show(), usa st.pyplot() o st.plotly_chart() según corresponda.
    También procura que los prints de los datos estén adaptados a ello para que se vean bien en la interfaz. Por ejemplo, para mostrar un DataFrame o series, usa st.dataframe() o st.table() en lugar de print().
    No pongas comentarios especificando que el código es para Streamlit, solo explicaciones del código si es necesario.
    Y de preferencia, usa librerías como Plotly para la visualización de datos.
    En el código, no incluyas la creación o importación del dataset, ya que el código interno del la aplicación lo cargará automáticamente.
    El nombre que llevará cada dataset siempre será el mismo que el del CSV, el cual se te indicará.
    
    Dependiendo de la solicitud del usuario, puedes no incluir la sección de código, pero siempre debes incluir la sección de resumen.
    Las dos secciones deberán estar separadas por la línea exacta (solamente eso, nada más):
    ########.
    No incluyas otros separadores como "Resumen:" o "Código:" en tu respuesta en ningun lugar.
    
    Si el usuario no proporciona una instrucción específica sobre lo que debes hacer (por ejemplo, que solo te dé un contexto del CSV), 
    deberás realizar una análisis exploratorio inicialmente.
"""

# ----- Funciones de procesamiento de datos -----
def read_csv_context(name, df, n_preview=5):
    # Extrae información general del CSV como columnas, filas, tipos de datos y valores faltantes
    cols = df.columns.tolist()
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.apply(lambda dt: dt.name).to_dict()
    missing = df.isnull().sum().to_dict()
    
    # Formatea la información en un resumen
    summary_lines = [
        f"**Dataset**: {name}",
        f"- Filas: {n_rows:,}",
        f"- Columnas: {n_cols:,}",
        "**Tipos de datos**:"
    ] + [f"  - {col}: {dtype}" for col, dtype in dtypes.items()] + [
        "**Valores faltantes**:"
    ] + [f"  - {col}: {cnt}" for col, cnt in missing.items()]

    # Estadísticas numéricas y primeras filas del dataset
    num_stats = df.select_dtypes(include=[np.number]).describe().round(3)
    preview = df.head(n_preview)
    
    return {
        "name": name,
        "columns": cols,
        "summary": "\n\n".join(summary_lines),
        "stats_df": num_stats,
        "preview_df": preview
    }

def process_uploaded_files(uploaded_files):
    # Procesa los archivos CSV subidos y construye contextos para cada archivo
    datasets = {}
    contexts = []
    error_files = []
    success_count = 0
    
    # Procesa cada archivo CSV subido
    for file in uploaded_files:
        try:
            name = os.path.splitext(file.name)[0]
            df = pd.read_csv(file)
            try:
                context = read_csv_context(name, df)
                datasets[name] = df
                contexts.append(context)
                success_count += 1
            except Exception as e:
                error_files.append(f"{name} (error al procesar el contexto: {str(e)})")
        except Exception as e:
            error_files.append(f"{file.name} (error al cargar: {str(e)})")
    
    return datasets, contexts, error_files, success_count

# ----- Funciones de integración con la IA/Chat -----
def build_initial_message(contexts, user_instruction):
    # Construye el mensaje inicial combinando los contextos del dataset y la instrucción del usuario
    parts = []
    for ctx in contexts:
        parts.append(f"Columnas en **{ctx['name']}**: " + ", ".join(ctx["columns"]))
        parts.append(ctx["summary"])
        parts.append(f"Primeras filas de **{ctx['name']}**:\n{ctx['preview_df'].to_csv(index=False)}")
    parts.append(f"Instrucción del usuario:\n{user_instruction}")
    return "\n\n".join(parts)

def init_chat():
    # Inicializa el chat configurando el cliente de GenAI con las instrucciones del sistema
    client = genai.Client(api_key=KEY)
    return client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=[types.Part.from_text(text=SYSTEM_INSTRUCTIONS)],
            # La temperatura controla la aleatoriedad de las respuestas generadas, el valor por defecto es 1.0
            # Un valor más bajo hará que las respuestas sean más predecibles y coherentes
            # Los valores bajos son ideales para el análisis de datos, donde se busca precisión
            temperature=0.5
        )
    )

def execute_ai_code(code, datasets):
    # Ejecuta el código generado por la IA en un entorno controlado, usando los datasets cargados
    exec_globals = {"np": np, "pd": pd, "st": st}
    for name, df in datasets.items():
        exec_globals[name] = df.copy()
    
    # Intercepta plt.show para integrarlo con Streamlit
    plt.show = lambda **kw: st.pyplot(plt.gcf())
    exec_globals["plt"] = plt
    
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    # Ejecuta el código en un entorno controlado
    try:
        exec(code, exec_globals)
        sys.stdout = old_stdout
        output = mystdout.getvalue().strip()
        return output, None
    except Exception as e:
        sys.stdout = old_stdout
        return None, str(e)

# ----- Funciones de la interfaz de usuario (UI) -----
def initialize_session_state():
    # Inicializa el estado de la sesión para mantener la información del chat, datasets, etc.
    if "chat" not in st.session_state: 
        st.session_state.chat = None
    if "history" not in st.session_state: 
        st.session_state.history = []
    if "contexts" not in st.session_state: 
        st.session_state.contexts = []
    if "datasets" not in st.session_state: 
        st.session_state.datasets = {}
    if "processing" not in st.session_state: 
        st.session_state.processing = False
    if "waiting_response" not in st.session_state: 
        st.session_state.waiting_response = False

def display_chat_history():
    # Muestra el historial del chat en la interfaz de Streamlit, incluyendo la ejecución de código
    for role, msg in st.session_state.history:
        if role == "Usuario":
            with st.chat_message(role.lower(), avatar="👨‍🔬"):
                st.markdown(msg)
        else:  # Insightify
            with st.chat_message(role.lower(), avatar="🧠"):
                # El marcador "########" indica el inicio del bloque de código
                if "########" in msg:
                    summary, code = msg.split("########", 1)
                    st.markdown(summary)
                    code = code.strip().strip("```python").strip("```")
                    st.code(code, language="python")
                    
                    # Ejecuta el código generado por la IA
                    with st.spinner('Ejecutando análisis...'):
                        output, error = execute_ai_code(code, st.session_state.datasets)
                        if error:
                            st.error(f"Error al ejecutar código: {error}")
                        elif output:
                            st.text(output)
                else:
                    st.markdown(msg)

def setup_page():
    # Configura la página principal y establece el logo y título del sitio
    st.set_page_config(page_title="Insightify", layout="wide", page_icon="logo.png")
    
    # Establece el logo y título
    # Necesario el markdown con código HTML para centrar bien el logo y el título
    st.markdown(
        "<div style='text-align: center;'><img src='data:image/jpeg;base64,{}' width='150px'/></div>".format(
            base64.b64encode(open("logo.png", "rb").read()).decode()
        ),
        unsafe_allow_html=True
    )
    st.markdown(
        "<h1 style='text-align: center; margin-left: 25px;'>Insightify</h1>",
        unsafe_allow_html=True
    )

# ----- Función principal de la aplicación -----
def main():
    # Configura la página, inicializa el estado y procesa los archivos CSV subidos
    setup_page()
    initialize_session_state()
    
    # Barral lateral para subir archivos CSV
    with st.sidebar:
        st.header("📂 Datasets")
        uploaded_files = st.file_uploader("Sube uno o más CSV", type="csv", accept_multiple_files=True)
    
    # Procesa los archivos CSV subidos y genera contextos
    if uploaded_files:
        with st.spinner('Procesando los archivos CSV...'):
            datasets, contexts, error_files, success_count = process_uploaded_files(uploaded_files)
            
            st.session_state.datasets = datasets
            st.session_state.contexts = contexts
            
            if error_files:
                st.error(f"No se pudieron procesar {len(error_files)} archivos: {', '.join(error_files)}")
            
            if success_count > 0:
                if st.session_state.chat is None:
                    st.session_state.chat = init_chat()
            else:
                st.warning("No se pudo cargar ningún dataset. Procura que sigan en formato CSV, no estén vacíos y sean consistentes.")
    
    # Muestra el historial del chat
    display_chat_history()
    
    # Procesa la respuesta de la IA si está pendiente
    if st.session_state.waiting_response and st.session_state.datasets:
        with st.spinner('Analizando tus datos...'):
            if len(st.session_state.history) == 1:  # Primera pregunta
                init_msg = build_initial_message(st.session_state.contexts, st.session_state.history[-1][1])
                resp = st.session_state.chat.send_message(init_msg)
            else:
                # Mensajes de seguimiento
                resp = st.session_state.chat.send_message(st.session_state.history[-1][1])
            
            st.session_state.history.append(("Insightify", resp.text))
        st.session_state.waiting_response = False
        st.session_state.processing = False
        st.rerun()
    
    # Entrada para el chat
    if st.session_state.datasets and not st.session_state.waiting_response:
        # Primera interacción con entrada centrada
        if not st.session_state.history:
            c1, c2, c3 = st.columns([1,3,1])
            with c2:
                user_input = st.chat_input("Escribe una instrucción", key="init", 
                                        disabled=st.session_state.processing)
        else:  # Interacciones posteriores
            user_input = st.chat_input("Haz otra pregunta…", disabled=st.session_state.processing)
        
        if user_input:
            st.session_state.history.append(("Usuario", user_input))
            st.session_state.waiting_response = True
            st.session_state.processing = True
            st.rerun()
    elif not uploaded_files:
        # Mensaje inicial si no se han subido archivos
        st.info("Sube al menos un CSV para comenzar.")

if __name__ == "__main__":
    main()
