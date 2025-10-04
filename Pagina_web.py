import streamlit as st
import pandas as pd
import joblib
import os

st.title("Prediction Application")

# File uploader (solo datos)
uploaded_data_file = st.file_uploader("Upload your Excel data file", type=['xlsx'])

data_df = None

if uploaded_data_file is not None:
    try:
        # Load the second sheet (index 1) of the Excel file
        data_df = pd.read_excel(uploaded_data_file, sheet_name=1)
        st.write("Data loaded successfully:")
        st.dataframe(data_df.head())
    except Exception as e:
        st.error(f"Error loading data file: {e}")

# Añadir aquí la lógica de preprocesamiento y predicción cuando corresponda.
