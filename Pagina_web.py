import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Prediction Application")

# =========================
# Utilidades de preprocesado
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
          .str.replace("/", "_", regex=False)
          .str.replace("-", "_", regex=False)
    )
    return df

def to_numeric_series(x):
    """
    Convierte strings con separadores de miles/espacios a número.
    Útil si en algún momento el valor viene como '1,234' o '1 234'.
    """
    if isinstance(x, (int, float, np.number)):
        return x
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").replace(" ", "")
    try:
        return float(x)
    except Exception:
        return np.nan

def align_to_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Alinea X a las features esperadas por el modelo.
    Agrega columnas faltantes con 0 y reordena.
    """
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "estimators_"):
        # e.g., Stacking/Ensemble con estimador final
        try:
            final_est = model.estimators_[-1]
            if hasattr(final_est, "feature_names_in_"):
                expected = list(final_est.feature_names_in_)
        except Exception:
            pass

    if expected is None:
        return X  # no hay info; lo dejamos tal cual y que el try/except de predicción capture
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    # Quita extras no esperadas por el modelo
    X = X[expected]
    return X

# =========================
# Carga de artefactos
# =========================
encoder_file_path = 'onehot_encoder.joblib'
scaler_file_path = 'minmax_scaler.joblib'
model_file_path = 'logistic_regression_best_model.joblib'

try:
    onehot_encoder = joblib.load(encoder_file_path)
    st.write("One-hot encoder loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: onehot_encoder.joblib not found at {encoder_file_path}. Please check the file path in your Google Drive.")
    st.stop()
except Exception as e:
    st.error(f"Error loading one-hot encoder: {e}")
    st.stop()

try:
    scaler = joblib.load(scaler_file_path)
    st.write("Min-Max scaler loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: minmax_scaler.joblib not found at {scaler_file_path}. Please check the file path in your Google Drive.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

try:
    best_model = joblib.load(model_file_path)
    st.write("Logistic Regression Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: logistic_regression_best_model.joblib not found at {model_file_path}. Please check the file path in your Google Drive.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =========================
# Entradas del usuario
# =========================
felder_input = st.selectbox(
    "Select Felder:",
    ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal']
)
examen_admision_input = st.number_input(
    "Enter Examen de admisión Universidad:",
    min_value=0.0, max_value=10.0, step=0.01
)

# Construye DataFrame de entrada
input_data = pd.DataFrame({
    'Felder': [str(felder_input)],
    'Examen_admisión_Universidad': [to_numeric_series(examen_admision_input)]
})
input_data = normalize_columns(input_data)

# =========================
# Preprocesamiento
# =========================
# 1) OneHot para 'Felder'
try:
    # Nos aseguramos de pasar 2-D (DataFrame) al encoder
    felder_df = input_data[['Felder']]

    # Si el encoder fue entrenado con handle_unknown='ignore', esto no fallará.
    # Si no, capturamos y explicamos el error.
    encoded_felder_input = onehot_encoder.transform(felder_df)

    # Columnas del encoder
    try:
        feature_names = onehot_encoder.get_feature_names_out(['Felder'])
    except Exception:
        # Fallback si no existe get_feature_names_out
        cats = getattr(onehot_encoder, "categories_", [None])[0]
        if cats is not None:
            feature_names = [f'Felder_{c}' for c in cats]
        else:
            feature_names = [f'Felder_{i}' for i in range(encoded_felder_input.shape[1])]

    encoded_felder_df_input = pd.DataFrame(encoded_felder_input.toarray() if hasattr(encoded_felder_input, "toarray") else encoded_felder_input,
                                           columns=feature_names)

    st.write("Felder encoded successfully.")
except Exception as e:
    st.error(
        "Error applying one-hot encoder. "
        "Si el valor seleccionado de 'Felder' no existía durante el entrenamiento "
        "y el encoder no tiene handle_unknown='ignore', fallará."
    )
    st.error(f"Detalle: {e}")
    st.stop()

# 2) Escalado de numéricas
numeric_columns_input = input_data.select_dtypes(include=['number']).columns.tolist()
scaled_df_input = pd.DataFrame()

if numeric_columns_input:
    try:
        data_to_scale = input_data[numeric_columns_input]
        scaled_data = scaler.transform(data_to_scale)
        scaled_df_input = pd.DataFrame(scaled_data, columns=numeric_columns_input)
        # Renombra para coincidir con pipelines típicos que esperan sufijo _scaled
        scaled_df_input.columns = [f"{c}_scaled" for c in scaled_df_input.columns]
        st.write("Numeric data scaled successfully.")
    except Exception as e:
        st.error(f"Error applying scaler: {e}")
        st.stop()

# 3) Combina todo
X_processed_input = pd.concat([scaled_df_input, encoded_felder_df_input], axis=1)
X_processed_input = align_to_model_features(X_processed_input, best_model)

# =========================
# Predicción
# =========================
if st.button("Predict"):
    try:
        prediction = best_model.predict(X_processed_input)
        st.subheader("Prediction:")
        st.write(str(prediction[0]))
    except ValueError as ve:
        st.error(f"ValueError during prediction: {ve}. This might be due to feature mismatch.")
        st.write("Processed input columns:", X_processed_input.columns.tolist())
        if hasattr(best_model, 'feature_names_in_'):
            st.write("Model expected features:", best_model.feature_names_in_.tolist())
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


