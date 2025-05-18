import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# === CARGAR LOS DATOS LIMPIOS DEL ETL ===
# Se asume que 'etl_carga_elastic.py' ya se ejecutó y generó 'matches_cleaned.json'
input_file_json = "matches_cleaned.json" 

try:
    # Leemos el archivo JSON generado por el script ETL
    df_filtrado = pd.read_json(input_file_json, lines=True)
    print(f"Datos JSON ('{input_file_json}') cargados exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{input_file_json}' no fue encontrado.")
    print("Asegúrate de que 'etl_carga_elastic.py' se haya ejecutado primero y haya generado este archivo.")
    exit()
except Exception as e:
    print(f"Error al cargar el archivo JSON '{input_file_json}': {e}")
    exit()

# El archivo JSON ya contiene 'match_result' y está filtrado por las ligas objetivo.
# También se han manejado los valores nulos básicos y duplicados en el ETL.

# === INGENIERÍA DE CARACTERÍSTICAS ADICIONALES (BASADAS EN FECHA) ===
# Convertir 'MatchDate' a datetime si no lo está (aunque el ETL ya lo formatea como YYYY-MM-DD)
df_filtrado['MatchDate'] = pd.to_datetime(df_filtrado['MatchDate'], errors='coerce')
df_filtrado = df_filtrado.dropna(subset=['MatchDate']) # Asegurar que no haya NaT después de la conversión

df_filtrado['match_dayofweek'] = df_filtrado['MatchDate'].dt.dayofweek 
df_filtrado['match_month'] = df_filtrado['MatchDate'].dt.month
df_filtrado['match_year'] = df_filtrado['MatchDate'].dt.year

# Columnas necesarias para el modelo (HomeTeam, AwayTeam, Division, match_result ya vienen del JSON)
# FTHome y FTAway se usan para *crear* el target 'match_result' en el ETL,
# pero no deben ser características de entrada para predecir un partido *futuro* donde el resultado es desconocido.
df_filtrado = df_filtrado.dropna(
    subset=['HomeTeam', 'AwayTeam', 'Division', 'match_result', 
            'match_dayofweek', 'match_month', 'match_year']
).copy()

print(f"Datos listos para ML. Total de partidos válidos: {len(df_filtrado)}")
print("Distribución de resultados:")
print(df_filtrado['match_result'].value_counts(normalize=True)) # normalize=True para ver porcentajes

# === SELECCIONAR CARACTERÍSTICAS Y VARIABLE OBJETIVO ===
# 'FTHome' y 'FTAway' se eliminan de las características porque no se conocerán antes de un partido futuro.
features = ['HomeTeam', 'AwayTeam', 'Division', 'match_dayofweek', 'match_month', 'match_year']
target = 'match_result'

X = df_filtrado[features]
y = df_filtrado[target]

print(f"\nCaracterísticas seleccionadas para el modelo: {features}")
print(f"Variable objetivo: {target}")

# === PREPROCESAMIENTO DE DATOS ===
categorical_features = ['HomeTeam', 'AwayTeam', 'Division']
# 'FTHome' y 'FTAway' eliminadas de numerical_features
numerical_features = ['match_dayofweek', 'match_month', 'match_year'] 

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features), # sparse_output=False puede ser útil para inspección
        ('num', 'passthrough', numerical_features) 
    ])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClases de resultados codificadas: {label_encoder.classes_} -> {np.unique(y_encoded)}")

# === DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nForma de los datos de entrenamiento (X_train): {X_train.shape}")
print(f"Forma de los datos de prueba (X_test): {X_test.shape}")

# === ENTRENAR UN MODELO DE CLASIFICACIÓN ===
# Se usa LogisticRegression como ejemplo. Puedes probar otros modelos.
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))]) # liblinear es bueno para datasets pequeños/medianos

print("\nEntrenando el modelo...")
model_pipeline.fit(X_train, y_train)
print(" Modelo entrenado exitosamente.")

# === EVALUAR EL MODELO ===
print("\nEvaluando el modelo en los datos de prueba...")
y_pred_encoded = model_pipeline.predict(X_test)

# Convertir predicciones y etiquetas de prueba de nuevo a sus nombres originales para el reporte
y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test_original, y_pred_original, zero_division=0))

accuracy = accuracy_score(y_test_original, y_pred_original)
print(f"Precisión (Accuracy): {accuracy:.4f}")

print("\n--- Matriz de Confusión ---")
# Asegurarse de que las etiquetas en la matriz de confusión estén en el orden correcto
conf_matrix = confusion_matrix(y_test_original, y_pred_original, labels=label_encoder.classes_)
print(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

# === PREDICCIÓN DE UN NUEVO PARTIDO (FUTURO) ===
print("\n--- Predicción de un nuevo partido (Ejemplo Futuro) ---")

# Crear una fecha futura para el partido de ejemplo
# Supongamos que queremos predecir un partido para el próximo sábado
from datetime import datetime, timedelta
today = datetime.today()
# Encontrar el próximo sábado
next_saturday = today + timedelta( (5-today.weekday()) % 7 ) 
if next_saturday < today: # Si hoy es sábado y ya pasó la hora del partido, tomar el siguiente.
    next_saturday += timedelta(days=7)

print(f"Fecha del partido de ejemplo: {next_saturday.strftime('%Y-%m-%d')}")

# Datos del nuevo partido. NO incluir FTHome, FTAway.
# Asegúrate de que los nombres de los equipos y la división existan en tus datos de entrenamiento
# o que OneHotEncoder(handle_unknown='ignore') los maneje adecuadamente.
new_match_data_dict = {
    'HomeTeam': 'Real Madrid',   # Ejemplo, usa un equipo de tus datos
    'AwayTeam': 'Barcelona',     # Ejemplo, usa un equipo de tus datos
    'Division': 'SP1',           # Ejemplo, usa una división de tus datos
    'match_dayofweek': next_saturday.weekday(), # Lunes=0, Domingo=6
    'match_month': next_saturday.month,
    'match_year': next_saturday.year
}
new_match_df = pd.DataFrame([new_match_data_dict])

# Asegurar que el DataFrame tenga las columnas en el mismo orden que 'features'
new_match_df = new_match_df[features] 

print("\nDatos del partido a predecir:")
print(new_match_df)

predicted_outcome_encoded = model_pipeline.predict(new_match_df)
predicted_outcome_proba = model_pipeline.predict_proba(new_match_df) # Obtener probabilidades

predicted_outcome_original = label_encoder.inverse_transform(predicted_outcome_encoded)

print(f"\nEl modelo predice el resultado para este partido: {predicted_outcome_original[0]}")

# Mostrar probabilidades
print("\nProbabilidades de cada resultado:")
for i, class_label in enumerate(label_encoder.classes_):
    print(f"  {class_label}: {predicted_outcome_proba[0][i]:.4f}")

# Puedes añadir más lógica aquí, como guardar la predicción, etc.

# Ejemplo de cómo obtener los nombres de las características después del OneHotEncoding
# Esto puede ser útil para entender la importancia de las características si usas modelos que la proporcionan
# try:
#     onehot_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
#     all_feature_names = np.concatenate([onehot_feature_names, numerical_features])
#     # print("\nNombres de todas las características después del preprocesamiento:")
#     # print(list(all_feature_names))
# except Exception as e:
#     print(f"No se pudieron obtener los nombres de las características del preprocesador: {e}")

