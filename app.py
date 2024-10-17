import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
import platform

# Inyectar CSS para cambiar las tipografías
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400&family=Lexend:wght@600&display=swap');

    h1, h2, h3 {
        font-family: 'Lexend', sans-serif;
    }

    p, div, label, span, input {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Muestra la versión de Python con la tipografía 'Inter'
st.write(f"<span>Versión de Python: {platform.python_version()}</span>", unsafe_allow_html=True)

# Cargar el modelo de Keras
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título con la tipografía Lexend
st.markdown("<h1>Reconocimiento de Imágenes</h1>", unsafe_allow_html=True)

# Cargar y mostrar imagen centrada
image = Image.open('OIG5.jpg')
st.image(image, width=350, use_column_width='always')

with st.sidebar:
    st.markdown("<h2>Usando un modelo entrenado</h2>", unsafe_allow_html=True)
    st.markdown("<p>Puedes usar esta app para identificar gestos.</p>", unsafe_allow_html=True)

# Entrada para tomar una foto
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # Convertir la imagen capturada a un formato adecuado
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar la predicción
    prediction = model.predict(data)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.markdown(f"<h2>Izquierda, con Probabilidad: {str(prediction[0][0])}</h2>", unsafe_allow_html=True)
    if prediction[0][1] > 0.5:
        st.markdown(f"<h2>Arriba, con Probabilidad: {str(prediction[0][1])}</h2>", unsafe_allow_html=True)
