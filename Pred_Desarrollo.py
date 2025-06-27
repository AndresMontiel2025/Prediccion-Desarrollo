# prompt: Haz todo el despliegue anterior en streamlit

import streamlit as st
import pandas as pd
import os
import joblib
import base64

# Function to load the dataset
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    return df

# Function to load the encoders and scaler
@st.cache_resource
def load_resources(path):
    os.chdir(path)
    onehot_encoder_comuna = joblib.load('onehot_encoder_comuna.pkl')
    onehot_encoder_prestacion_servicio = joblib.load('onehot_encoder_prestacion_servicio.pkl')
    scaler = joblib.load('scaler_gestion_academica.pkl')
    gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')
    feature_names = joblib.load('gradient_boosting_features.pkl')
    ordinal_encoder_desarrollo = joblib.load('ordinal_encoder_desarrollo.pkl')
    return (onehot_encoder_comuna, onehot_encoder_prestacion_servicio, scaler,
            gradient_boosting_model, feature_names, ordinal_encoder_desarrollo)

# Streamlit App Title
st.title("Predicción del Desarrollo Educativo")

# Path to your files in Google Drive
# IMPORTANT: You will need to mount Google Drive or make these files
# accessible to the environment where Streamlit is running.
# For a deployed Streamlit app, you'd typically use cloud storage
# or include the files in your application's directory.
# For local testing with Colab, you might manually copy files.
#DATASET_PATH = "/content/drive/MyDrive/Inteligencia Analítica/Trabajo Final/Dataset/Conjunto de datos nuevos.xlsx"
#MODELS_PATH = "/content/drive/MyDrive/Inteligencia Analítica/Trabajo Final/Modelos_Guardados"

try:
    # Load data
    df = load_data(DATASET_PATH)
    df_original_for_display = df.copy() # Keep a copy for displaying original columns

    # Load models and resources
    onehot_encoder_comuna, onehot_encoder_prestacion_servicio, scaler, \
    gradient_boosting_model, feature_names, ordinal_encoder_desarrollo = load_resources(MODELS_PATH)

    st.write("Datos cargados y modelos/recursos listos.")

    # --- Data Preprocessing (as in the notebook) ---
    df = df.drop(columns=['año', 'codigo_dane', 'establecimiento educativo'])

    # Ensure categorical columns are of type 'category'
    df['prestacion_servicio'] = df['prestacion_servicio'].astype('category')
    df['comuna_establecimiento'] = df['comuna_establecimiento'].astype('category')

    # Apply one-hot encoding
    df_comuna_encoded = onehot_encoder_comuna.transform(df[['comuna_establecimiento']])
    df_ps_encoded = onehot_encoder_prestacion_servicio.transform(df[['prestacion_servicio']])

    # Convert to DataFrames
    df_comuna_encoded_df = pd.DataFrame(df_comuna_encoded, columns=onehot_encoder_comuna.get_feature_names_out(['comuna_establecimiento']))
    df_ps_encoded_df = pd.DataFrame(df_ps_encoded, columns=onehot_encoder_prestacion_servicio.get_feature_names_out(['prestacion_servicio']))

    # Concatenate and drop original categorical columns
    df = pd.concat([df.drop(columns=['comuna_establecimiento', 'prestacion_servicio']).reset_index(drop=True),
                      df_comuna_encoded_df.reset_index(drop=True),
                      df_ps_encoded_df.reset_index(drop=True)], axis=1)

    # Scale 'gestion_academica'
    df['gestion_academica_scaled'] = scaler.transform(df[['gestion_academica']]) # Use transform as scaler is already fitted

    # --- Prediction ---
    # Ensure the input DataFrame has the same columns and order as used during training
    # Drop 'gestion_academica' as we will use the scaled version
    if 'gestion_academica' in df.columns:
        df_model_input = df.drop(columns=['gestion_academica'])
    else:
         df_model_input = df.copy()

    # Select only the features the model was trained on
    df_model_input = df_model_input[feature_names].copy()

    # Make predictions
    predictions_encoded = gradient_boosting_model.predict(df_model_input)

    # Decode predictions
    predictions_original = ordinal_encoder_desarrollo.inverse_transform(predictions_encoded.reshape(-1, 1))

    # Add predictions to the original DataFrame (without encoding)
    df_original_for_display['prediccion_desarrollo'] = predictions_original.flatten()

    # --- Display Results ---
    st.subheader("Resultados de la Predicción")

    # Display the relevant columns from the original DataFrame + prediction
    display_cols = [col for col in df_original_for_display.columns if col not in ['año', 'codigo_dane', 'establecimiento educativo']]
    display_cols.append('prediccion_desarrollo')

    st.dataframe(df_original_for_display[display_cols])

    # Option to download the results
    csv_export = df_original_for_display[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar resultados como CSV",
        data=csv_export,
        file_name='predicciones_desarrollo.csv',
        mime='text/csv',
    )


except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos (dataset y modelos) existen en las rutas especificadas en Google Drive.")
    st.info("Para ejecutar esto localmente, necesitarás copiar los archivos del dataset y modelos a un directorio accesible por tu aplicación Streamlit.")
except Exception as e:
    st.error(f"Ocurrió un error: {e}")
    st.error("Por favor, verifica que los archivos de modelos y encoders son correctos y coinciden con el entrenamiento.")

# Instructions on how to run
st.sidebar.subheader("Cómo ejecutar este código")
st.sidebar.write("""
1.  **Guarda** el código anterior como un archivo Python (ej: `app.py`).
2.  **Asegúrate** de tener `streamlit`, `pandas`, `joblib`, `openpyxl`, `scikit-learn` instalados (`pip install streamlit pandas joblib openpyxl scikit-learn`).
3.  **Coloca** el archivo de dataset (`Conjunto de datos nuevos.xlsx`) y la carpeta `Modelos_Guardados` (con todos los `.pkl` dentro) en rutas accesibles por donde ejecutas Streamlit. Si usas Google Colab para desarrollar y luego quieres desplegar, deberás descargar estos archivos de Drive. Para un despliegue web real, súbelos a un bucket S3, Google Cloud Storage, o inclúyelos en tu repositorio si no son demasiado grandes.
4.  **Actualiza** las variables `DATASET_PATH` y `MODELS_PATH` en el código para que apunten a las ubicaciones correctas de tus archivos.
5.  **Ejecuta** la aplicación desde tu terminal: `streamlit run app.py`
""")
