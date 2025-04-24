import streamlit as st
from Insightify import read_csv_context, build_initial_message, instructions
from google import genai
from google.genai import types
import os
from PIL import Image
import base64
from io import BytesIO
import pandas as pd

KEY = "tu llave del api xd"
client = genai.Client(api_key=KEY)
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(text=instructions)],
        temperature=1.0,
    )
)

st.set_page_config(page_title="Insightify", page_icon="logo.jpg")

def get_image_base64(img_path, resize_width):
    img = Image.open(img_path)
    w_percent = (resize_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((resize_width, h_size))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str 
    
col1, col2, col3 = st.columns([1, 2, 1])
with col2:

    img_base64 = get_image_base64("logo.jpg", resize_width=100)  
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src="data:image/jpeg;base64,{img_base64}" alt="logo" />
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h2 style='text-align: center;'>Insightify</h2>", unsafe_allow_html=True)


if "history" not in st.session_state:
    st.session_state.history = []

# se sube el archivo CSV en el silider *opcional
st.sidebar.markdown("### Subir archivo CSV")
csv_file = st.sidebar.file_uploader("Selecciona un archivo", type=["csv"])


# se prueba si ya se ha subido un archivo desde el slider  
if csv_file is not None:
    with open("temp.csv", "wb") as f:
        f.write(csv_file.read())
    file_to_use = "temp.csv"
    st.success("Archivo subido correctamente.")
else:   #si el usuario no sube un archivo se usara el archivo por defecto que esta en la carpeta datasets
    file_to_use = "datasets/csv1.csv"
    st.info("No has subido un archivo. Se usará el archivo por defecto.")



user_instruction = st.chat_input("Escribe tu instrucción aquí") #cuadrito de texto xd

# entrada
if user_instruction:
    context = read_csv_context(file_to_use)
    init_msg = build_initial_message([context], user_instruction)

    # burbuja
    with st.chat_message("user"):
        st.write(user_instruction)
    st.session_state.history.append(("user", user_instruction))

    # Generar respuesta
    with st.spinner("Analizando..."):
        try:
            response = chat.send_message(init_msg)
            with st.chat_message("assistant"):
                st.write(response.text)
            st.session_state.history.append(("assistant", response.text))
        except Exception as e:
            st.error(f"Error al generar la respuesta: {e}")

# Mostrar historial de conversación
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Historial de conversación")
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.write(msg)