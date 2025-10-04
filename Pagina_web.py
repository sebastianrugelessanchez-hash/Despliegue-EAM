%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

encoder = joblib.load('onehot_encoder.joblib')
scaler = joblib.load('minmax_scaler.joblib')
model = joblib.load('logistic_regression_best_model.joblib')

st.title('Course Approval Prediction')

# Define the main app function
def main():
    st.header('Input Student Data')

    # Load the data
    excel_file_path = "Aprobacion curso 2019.xlsx"
    try:
        df = pd.read_excel(excel_file_path, sheet_name=1) # Assuming index 1 for the second sheet
    except ValueError:
        st.error("Could not load the second sheet from the Excel file.")
        return


    df = df.drop(columns=['ID', 'Año - Semestre'])

  
    try:
        felder_encoded = encoder.transform(df[['Felder']])
        felder_encoded_df = pd.DataFrame(felder_encoded, columns=encoder.get_feature_names_out(['Felder']))
        df = pd.concat([df.drop('Felder', axis=1), felder_encoded_df], axis=1)
    except Exception as e:
        st.error(f"Error during one-hot encoding: {e}")
        return


    try:
        df[['Examen_admisión']] = scaler.transform(df[['Examen_admisión']])
        df = df.rename(columns={'Examen_admisión': 'Examen_admisión_Universidad_scaled'})
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return


    try:
        expected_features = model.feature_names_in_
        df_ordered = df[expected_features]
    except Exception as e:
        st.error(f"Error reordering columns: {e}")
        return


    predictions = model.predict(df_ordered)

    # Display the predictions
    st.header('Prediction Results')
    st.write('The predictions for course approval are:')
    st.write(predictions)


if __name__ == '__main__':
    main()