# prompt: Haz todo el despliegue anterior en streamlit. Se va a subir a github y se desplegará en streamlit. los archivos .pkl se encontrarán en una carpeta llamada Modelos_Guardados junto con el archivo .py. Las predicciones se harán con un archivo excel a subir

import streamlit as st
import pandas as pd
import joblib
import os

# Function to load models and encoders
@st.cache_resource
def load_models_and_encoders():
  """Loads the trained models, encoders, and scaler."""
  model_path = 'Modelos_Guardados/'
  try:
    onehot_encoder_comuna = joblib.load(os.path.join(model_path, 'onehot_encoder_comuna.pkl'))
    onehot_encoder_prestacion_servicio = joblib.load(os.path.join(model_path, 'onehot_encoder_prestacion_servicio.pkl'))
    scaler = joblib.load(os.path.join(model_path, 'scaler_gestion_academica.pkl'))
    gradient_boosting_model = joblib.load(os.path.join(model_path, 'gradient_boosting_model.pkl'))
    feature_names = joblib.load(os.path.join(model_path, 'gradient_boosting_features.pkl'))
    ordinal_encoder_desarrollo = joblib.load(os.path.join(model_path, 'ordinal_encoder_desarrollo.pkl'))
    return (onehot_encoder_comuna, onehot_encoder_prestacion_servicio, scaler,
            gradient_boosting_model, feature_names, ordinal_encoder_desarrollo)
  except FileNotFoundError as e:
    st.error(f"Error loading model files. Please ensure the 'Modelos_Guardados' directory and its contents are present. Missing file: {e}")
    return None

# Load models and encoders
loaded_resources = load_models_and_encoders()

if loaded_resources is None:
  st.stop() # Stop the app if resources didn't load

onehot_encoder_comuna, onehot_encoder_prestacion_servicio, scaler, gradient_boosting_model, feature_names, ordinal_encoder_desarrollo = loaded_resources


st.title("Predicción del Desarrollo Institucional")

st.write("Esta aplicación predice el nivel de desarrollo institucional basándose en datos de instituciones educativas.")

uploaded_file = st.file_uploader("Sube tu archivo Excel para la predicción", type=["xlsx"])

if uploaded_file is not None:
  try:
    df_original = pd.read_excel(uploaded_file)
    st.write("Archivo cargado exitosamente:")
    st.dataframe(df_original.head())

    # Data Preprocessing (as in your Colab notebook)
    df = df_original.copy()

    # Eliminar las columnas especificadas
    columns_to_drop = ['año', 'codigo_dane', 'establecimiento educativo']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Convert categorical columns (handle potential missing columns gracefully)
    if 'prestacion_servicio' in df.columns:
        df['prestacion_servicio'] = df['prestacion_servicio'].astype('category')
    if 'comuna_establecimiento' in df.columns:
        df['comuna_establecimiento'] = df['comuna_establecimiento'].astype('category')

    # Apply encoders (handle potential missing columns gracefully)
    df_comuna_encoded_df = pd.DataFrame()
    if 'comuna_establecimiento' in df.columns:
      df_comuna_encoded = onehot_encoder_comuna.transform(df[['comuna_establecimiento']])
      df_comuna_encoded_df = pd.DataFrame(df_comuna_encoded.toarray(), columns=onehot_encoder_comuna.get_feature_names_out(['comuna_establecimiento']))
      df = df.drop(columns=['comuna_establecimiento']).reset_index(drop=True)
      df = pd.concat([df.reset_index(drop=True), df_comuna_encoded_df.reset_index(drop=True)], axis=1)

    df_ps_encoded_df = pd.DataFrame()
    if 'prestacion_servicio' in df.columns:
      df_ps_encoded = onehot_encoder_prestacion_servicio.transform(df[['prestacion_servicio']])
      df_ps_encoded_df = pd.DataFrame(df_ps_encoded.toarray(), columns=onehot_encoder_prestacion_servicio.get_feature_names_out(['prestacion_servicio']))
      df = df.drop(columns=['prestacion_servicio']).reset_index(drop=True)
      df = pd.concat([df.reset_index(drop=True), df_ps_encoded_df.reset_index(drop=True)], axis=1)


    # Scale 'gestion_academica' (handle potential missing column)
    if 'gestion_academica' in df.columns:
        df['gestion_academica_scaled'] = scaler.transform(df[['gestion_academica']])
        # Drop original gestion_academica if the model only expects the scaled version
        # Check feature_names to confirm
        if 'gestion_academica' in df.columns and 'gestion_academica' not in feature_names:
             df = df.drop(columns=['gestion_academica'])

    # Ensure the DataFrame has the same columns as the training data
    # This is crucial for the model to work correctly.
    # Create a dataframe with all expected columns, filled with 0 initially
    df_model_input = pd.DataFrame(0, index=df.index, columns=feature_names)

    # Copy the matching columns from the processed dataframe
    for col in feature_names:
        if col in df.columns:
            df_model_input[col] = df[col]
        else:
            # If a feature is missing in the uploaded data but was in training,
            # it will remain 0 as initialized, which is typically how one-hot encoded
            # missing categories are handled. For other features, this might need
            # imputation depending on how the original training data was handled.
            # Assuming missing features in the input should be treated as 0 based
            # on the one-hot encoding logic.
            pass


    # Perform predictions
    predictions_encoded = gradient_boosting_model.predict(df_model_input)

    # Decode predictions
    predictions_original = ordinal_encoder_desarrollo.inverse_transform(predictions_encoded.reshape(-1, 1))

    # Add predictions to the original dataframe for display
    df_original['prediccion_desarrollo'] = predictions_original.flatten()

    st.subheader("Resultados de la Predicción")
    st.dataframe(df_original)

  except Exception as e:
    st.error(f"Ocurrió un error durante el procesamiento: {e}")
