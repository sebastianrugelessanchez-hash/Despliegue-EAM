import streamlit as st
import numpy as np
import pickle

# -------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------
st.set_page_config(
    page_title="Course Approval Prediction",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align: center;'>Course Approval Prediction</h1>", unsafe_allow_html=True)

# -------------------------------
# CARGA DEL MODELO (OPCIONAL)
# -------------------------------
# Si tienes un modelo entrenado, descomenta esto:
# with open("modelo_prediccion.pkl", "rb") as file:
#     model = pickle.load(file)

# -------------------------------
# INTERFAZ DE USUARIO
# -------------------------------
st.write("Select Felder category:")
felder_category = st.selectbox("Select Felder category:", ["activo", "reflexivo", "sensorial", "intuitivo", "visual", "verbal", "secuencial", "global"], label_visibility="collapsed")

st.write("Enter Examen de admisión score:")
exam_score = st.number_input("Enter Examen de admisión score:", min_value=0.0, max_value=10.0, step=0.1, format="%.2f", label_visibility="collapsed")

# -------------------------------
# PREDICCIÓN (SIMULADA)
# -------------------------------
# Si no tienes modelo, se usa una fórmula de ejemplo:
# (solo para demostrar la interfaz)
def predict_score(category, exam):
    # Ejemplo simple: diferente peso según tipo de aprendizaje
    weights = {
        "activo": 0.56, "reflexivo": 0.59, "sensorial": 0.52, "intuitivo": 0.55,
        "visual": 0.54, "verbal": 0.50, "secuencial": 0.57, "global": 0.60
    }
    base = weights.get(category, 0.55)
    return round(base * exam + np.random.uniform(1, 3), 4)

if exam_score > 0:
    predicted_score = predict_score(felder_category, exam_score)
    st.markdown(f"**Predicted Course Approval Score:** <span style='color:#4ade80; font-size:18px;'>{predicted_score}</span>", unsafe_allow_html=True)

# -------------------------------
# INSTRUCCIONES
# -------------------------------
st.markdown("""
This application predicts the likelihood of course approval based on your Felder learning style and university entrance exam score.

### How to use:
1. Select your Felder learning style from the dropdown menu.  
2. Enter your university entrance exam score in the number input field.  
3. The predicted course approval score will be displayed below.  

The predicted score is a numerical value indicating the estimated approval score. Higher values generally suggest a higher likelihood of course approval.
""")

