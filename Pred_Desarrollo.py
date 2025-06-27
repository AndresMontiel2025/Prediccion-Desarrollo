# prompt: Haz todo el despliegue anterior en streamlit. Se va a subir a github y se desplegará en streamlit

import streamlit as st
import pandas as pd
import joblib
import os

# Título de la aplicación
st.title("Predicción del Desarrollo del Establecimiento Educativo")

# Definir las rutas de los archivos. Asume que los archivos están en el mismo directorio
# o en subdirectorios relativos al script de Streamlit.
# Es mejor NO usar rutas absolutas que dependen de Google Drive en Streamlit.
# Deberás subir estos archivos (modelo, encoders, scaler, features, datos) a tu repositorio de GitHub.
MODEL_PATH = 'Modelos_Guardados/gradient_boosting_model.pkl'
FEATURES_PATH = 'Modelos_Guardados/gradient_boosting_features.pkl'
ONEHOT_COMUNA_PATH = 'Modelos_Guardados/onehot_encoder_comuna.pkl'
ONEHOT_PS_PATH = 'Modelos_Guardados/onehot_encoder_prestacion_servicio.pkl'
SCALER_PATH = 'Modelos_Guardados/scaler_gestion_academica.pkl'
ORDINAL_DESARROLLO_PATH = 'ordinal_encoder_desarrollo.pkl'
# Si el archivo de datos original es necesario para mostrar las entradas, también cárgalo.
# Asegúrate de que este archivo 'Conjunto de datos nuevos.xlsx' también esté en tu repo.
DATA_PATH = 'Conjunto de datos nuevos.xlsx'


# Cargar los modelos y objetos necesarios
@st.cache_resource # Cachea la carga para no recargar en cada interacción
def load_resources():
    try:
        gradient_boosting_model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        onehot_encoder_comuna = joblib.load(ONEHOT_COMUNA_PATH)
        onehot_encoder_prestacion_servicio = joblib.load(ONEHOT_PS_PATH)
        scaler = joblib.load(SCALER_PATH)
        ordinal_encoder_desarrollo = joblib.load(ORDINAL_DESARROLLO_PATH)
        return (gradient_boosting_model, feature_names, onehot_encoder_comuna,
                onehot_encoder_prestacion_servicio, scaler, ordinal_encoder_desarrollo)
    except FileNotFoundError as e:
        st.error(f"Error al cargar recursos: {e}. Asegúrate de que todos los archivos (.pkl, .xlsx) estén en las rutas correctas en tu repositorio.")
        return None

resources = load_resources()

if resources:
    (gradient_boosting_model, feature_names, onehot_encoder_comuna,
     onehot_encoder_prestacion_servicio, scaler, ordinal_encoder_desarrollo) = resources

    st.write("Recursos cargados exitosamente.")

    # Opción para subir un archivo Excel con nuevos datos (opcional)
    uploaded_file = st.file_uploader("Sube un archivo Excel con los datos a predecir", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            st.subheader("Datos cargados (primeras 5 filas):")
            st.dataframe(df.head())

            # --- Preprocesamiento de los datos subidos ---
            # Asumiendo que la estructura de columnas del archivo subido es similar al original
            # y que necesita el mismo preprocesamiento.

            # Eliminar columnas no necesarias (si existen)
            cols_to_drop = ['año', 'codigo_dane', 'establecimiento educativo']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

            # Convertir a categóricas (si existen)
            if 'prestacion_servicio' in df.columns:
                df['prestacion_servicio'] = df['prestacion_servicio'].astype('category')
            if 'comuna_establecimiento' in df.columns:
                 df['comuna_establecimiento'] = df['comuna_establecimiento'].astype('category')


            # Aplicar One-Hot Encoding (usando los encoders cargados)
            # Asegúrate de manejar casos donde las columnas categóricas no estén presentes
            df_encoded_list = []
            df_remaining = df.copy()

            if 'comuna_establecimiento' in df_remaining.columns:
                # Manejar valores desconocidos: 'ignore' asigna ceros
                df_comuna_encoded = onehot_encoder_comuna.transform(df_remaining[['comuna_establecimiento']])
                df_comuna_encoded_df = pd.DataFrame(df_comuna_encoded, columns=onehot_encoder_comuna.get_feature_names_out(['comuna_establecimiento']), index=df_remaining.index)
                df_encoded_list.append(df_comuna_encoded_df)
                df_remaining = df_remaining.drop(columns=['comuna_establecimiento'])

            if 'prestacion_servicio' in df_remaining.columns:
                 # Manejar valores desconocidos: 'ignore' asigna ceros
                df_ps_encoded = onehot_encoder_prestacion_servicio.transform(df_remaining[['prestacion_servicio']])
                df_ps_encoded_df = pd.DataFrame(df_ps_encoded, columns=onehot_encoder_prestacion_servicio.get_feature_names_out(['prestacion_servicio']), index=df_remaining.index)
                df_encoded_list.append(df_ps_encoded_df)
                df_remaining = df_remaining.drop(columns=['prestacion_servicio'])


            # Concatenar las columnas codificadas y las columnas restantes
            df_processed = pd.concat([df_remaining] + df_encoded_list, axis=1)


            # Aplicar Scaling a 'gestion_academica' (si existe)
            if 'gestion_academica' in df_processed.columns:
                 # Usar transform, no fit_transform, ya que el scaler ya fue entrenado
                 df_processed['gestion_academica_scaled'] = scaler.transform(df_processed[['gestion_academica']])
                 # Opcional: eliminar la columna original si solo se necesita la escalada
                 # df_processed = df_processed.drop(columns=['gestion_academica'])


            # Asegurarse de que el DataFrame procesado tenga exactamente las mismas columnas
            # y en el mismo orden que las características esperadas por el modelo
            # Esto es CRUCIAL para evitar errores de predicción.
            # Rellenar con 0 si faltan columnas (pueden faltar columnas de one-hot encoding si
            # en el archivo subido no están todas las categorías presentes en el entrenamiento)
            # Asegurarse de que solo están las columnas esperadas y en el orden correcto
            df_model_input = pd.DataFrame(index=df_processed.index)
            for col in feature_names:
                 if col in df_processed.columns:
                     df_model_input[col] = df_processed[col]
                 else:
                     # Si una columna esperada por el modelo no está en los datos subidos,
                     # probablemente es una categoría de one-hot encoding que no apareció.
                     # Se rellena con 0, como lo haría transform(handle_unknown='ignore').
                     df_model_input[col] = 0

            # --- Realizar predicciones ---
            if not df_model_input.empty:
                 # Realizar predicciones (valores codificados)
                 predictions_encoded = gradient_boosting_model.predict(df_model_input)

                 # Decodificar las predicciones a sus valores originales
                 predictions_original = ordinal_encoder_desarrollo.inverse_transform(predictions_encoded.reshape(-1, 1))

                 # Agregar la columna de predicciones decodificadas al DataFrame original cargado
                 # (para mostrar las entradas originales junto a la predicción)
                 df['prediccion_desarrollo'] = predictions_original.flatten()

                 st.subheader("Resultados de la Predicción:")
                 # Mostrar el DataFrame original con la nueva columna de predicciones
                 st.dataframe(df[['codigo_dane', 'establecimiento educativo', 'prestacion_servicio',
                                  'comuna_establecimiento', 'gestion_academica', 'prediccion_desarrollo']])

            else:
                 st.warning("El DataFrame de entrada para el modelo está vacío después del preprocesamiento.")


        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento del archivo: {e}")

    else:
        st.info("Por favor, sube un archivo Excel para obtener predicciones.")

    # Opcional: Mostrar las predicciones del archivo original usado en Colab (si lo cargas)
    # Esto es útil para verificar que la carga y predicción funcionan igual que en Colab
    # if st.checkbox("Mostrar predicciones del archivo de entrenamiento original"):
    #     try:
    #         df_original_train = pd.read_excel(DATA_PATH)
    #         # Aquí replicarías el preprocesamiento y predicción para df_original_train
    #         # similar a lo que hiciste en Colab y lo mostrarías.
    #         # (Este código sería más extenso y no se incluye aquí para brevedad,
    #         # pero seguiría la lógica de preprocesamiento y predicción definida arriba).
    #         st.write("Implementar carga y predicción para el archivo de entrenamiento original.")
    #     except FileNotFoundError:
    #         st.warning(f"No se encontró el archivo de datos original en la ruta: {DATA_PATH}")

else:
    st.error("La aplicación no pudo cargar los recursos necesarios. Consulta los logs para más detalles.")

