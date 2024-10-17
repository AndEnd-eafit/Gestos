import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write(f"<span style='font-family:Lexend;'>Versión de Python: {platform.python_version()}</span>", unsafe_allow_html=True)

# Cargar el modelo de Keras
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título con la tipografía Lexend
st.markdown("<h1 style='font-family:Lexend;'>Reconocimiento de Imágenes</h1>", unsafe_allow_html=True)

# Cargar y mostrar imagen centrada
image = Image.open('OIG5.jpg')
st.image(image, width=350, use_column_width='always')

with st.sidebar:
    st.markdown("<h2 style='font-family:Lexend;'>Usando un modelo entrenado</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:Inter;'>Puedes usar esta app para identificar gestos.</p>", unsafe_allow_html=True)

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
        st.markdown(f"<h2 style='font-family:Lexend;'>Izquierda, con Probabilidad: {str(prediction[0][0])}</h2>", unsafe_allow_html=True)
    if prediction[0][1] > 0.5:
        st.markdown(f"<h2 style='font-family:Lexend;'>Arriba, con Probabilidad: {str(prediction[0][1])}</h2>", unsafe_allow_html=True)

    # st.header('Derecha, con Probabilidad: '+str( prediction[0][2]))


