import streamlit as st
import pandas as pd
import joblib
import os


st.title("Prediction Application")

# Load the preprocessor and model files
# Mount Google Drive to access the files


encoder_file_path = 'onehot_encoder.joblib'
scaler_file_path = 'minmax_scaler.joblib'
model_file_path = 'logistic_regression_best_model.joblib'


try:
    onehot_encoder = joblib.load(encoder_file_path)
    st.write("One-hot encoder loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: onehot_encoder.joblib not found at {encoder_file_path}. Please check the file path in your Google Drive.")
    st.stop() # Stop the app if file is not found
except Exception as e:
    st.error(f"Error loading one-hot encoder: {e}")
    st.stop()

try:
    scaler = joblib.load(scaler_file_path)
    st.write("Min-Max scaler loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: minmax_scaler.joblib not found at {scaler_file_path}. Please check the file path in your Google Drive.")
    st.stop() # Stop the app if file is not found
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

try:
    best_model = joblib.load(model_file_path)
    st.write("Logistic Regression Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: logistic_regression_best_model.joblib not found at {model_file_path}. Please check the file path in your Google Drive.")
    st.stop() # Stop the app if file is not found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Input fields for user
felder_input = st.selectbox("Select Felder:", ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal']) # Add all possible 'Felder' categories
examen_admision_input = st.number_input("Enter Examen de admisión Universidad:", min_value=0.0, max_value=10.0, step=0.01) # Adjust max_value and step as needed

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'Felder': [felder_input],
    'Examen_admisión_Universidad': [examen_admision_input]
})

# Ensure 'Felder' is treated as string for encoding
input_data['Felder'] = input_data['Felder'].astype(str)

# --- Preprocessing Steps ---

# Apply the one-hot encoder
try:
    felder_data_input = input_data[['Felder']]
    encoded_felder_input = onehot_encoder.transform(felder_data_input)

    if hasattr(onehot_encoder, 'get_feature_names_out'):
        encoded_felder_df_input = pd.DataFrame(encoded_felder_input, columns=onehot_encoder.get_feature_names_out(['Felder']))
    else:
         try:
            feature_names = onehot_encoder.categories_[0]
            encoded_felder_df_input = pd.DataFrame(encoded_felder_input, columns=[f'Felder_{name}' for name in feature_names])
         except:
            # Fallback if feature names cannot be retrieved
            encoded_felder_df_input = pd.DataFrame(encoded_felder_input, columns=[f'Felder_{i}' for i in range(encoded_felder_input.shape[1])])


    st.write("Felder encoded successfully.")

except Exception as e:
    st.error(f"Error applying one-hot encoder: {e}")
    st.stop()


# Select the numeric columns for scaling
numeric_columns_input = input_data.select_dtypes(include=['number']).columns.tolist()

scaled_df_input = pd.DataFrame() # Initialize an empty DataFrame for scaled data

if numeric_columns_input:
    data_to_scale_input = input_data[numeric_columns_input]

    # Apply the scaler
    try:
        scaled_data_input = scaler.transform(data_to_scale_input)
        scaled_df_input = pd.DataFrame(scaled_data_input, columns=numeric_columns_input)
        scaled_df_input.columns = [f'{col}_scaled' for col in scaled_df_input.columns] # Rename to match expected

        st.write("Numeric data scaled successfully.")

    except Exception as e:
        st.error(f"Error applying scaler: {e}")
        st.stop()

# Combine encoded and scaled data
X_processed_input = pd.concat([scaled_df_input, encoded_felder_df_input], axis=1)

# --- Prediction ---
if st.button("Predict"):
    try:
        # Ensure column order matches the model's expected feature names
        if hasattr(best_model, 'feature_names_in_'):
            expected_features = best_model.feature_names_in_
            # Add any missing columns from expected features to X_processed_input with 0
            for col in expected_features:
                if col not in X_processed_input.columns:
                    X_processed_input[col] = 0
            # Reorder columns
            X_processed_input = X_processed_input[expected_features]

        elif hasattr(best_model, 'estimators_'):
             try:
                final_estimator = best_model.estimators_[-1]
                if hasattr(final_estimator, 'feature_names_in_'):
                    expected_features = final_estimator.feature_names_in_
                    for col in expected_features:
                        if col not in X_processed_input.columns:
                            X_processed_input[col] = 0
                    X_processed_input = X_processed_input[expected_features]
             except:
                 st.warning("Could not access feature names from the model's final estimator. Prediction might fail due to feature mismatch.")
        else:
            st.warning("Could not access feature names from the model. Prediction might fail due to feature mismatch.")


        # Make prediction
        prediction = best_model.predict(X_processed_input)

        st.subheader("Prediction:")
        # Display the prediction as "si" or "no"
        st.write(str(prediction[0]))

    except ValueError as ve:
        st.error(f"ValueError during prediction: {ve}. This might be due to feature mismatch.")
        st.write("Processed input columns:", X_processed_input.columns.tolist())
        if hasattr(best_model, 'feature_names_in_'):
             st.write("Model expected features:", best_model.feature_names_in_.tolist())
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
