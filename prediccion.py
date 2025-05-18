import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

input_file_json = "matches_cleaned.json" 

try:
    df_filtrado = pd.read_json(input_file_json, lines=True)
    print(f"Datos JSON ('{input_file_json}') cargados exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{input_file_json}' no fue encontrado.")
    print("Asegúrate de que 'etl_carga_elastic.py' se haya ejecutado primero y haya generado este archivo.")
    exit()
except Exception as e:
    print(f"Error al cargar el archivo JSON '{input_file_json}': {e}")
    exit()

df_filtrado['MatchDate'] = pd.to_datetime(df_filtrado['MatchDate'], errors='coerce')
df_filtrado = df_filtrado.dropna(subset=['MatchDate']) 
df_filtrado['match_dayofweek'] = df_filtrado['MatchDate'].dt.dayofweek 
df_filtrado['match_month'] = df_filtrado['MatchDate'].dt.month
df_filtrado['match_year'] = df_filtrado['MatchDate'].dt.year

df_filtrado = df_filtrado.dropna(
    subset=['HomeTeam', 'AwayTeam', 'Division', 'match_result', 
            'match_dayofweek', 'match_month', 'match_year']
).copy()

print(f"Datos listos para ML. Total de partidos válidos: {len(df_filtrado)}")
print("Distribución de resultados:")
print(df_filtrado['match_result'].value_counts(normalize=True))

features = ['HomeTeam', 'AwayTeam', 'Division', 'match_dayofweek', 'match_month', 'match_year']
target = 'match_result'

X = df_filtrado[features]
y = df_filtrado[target]

print(f"\nCaracterísticas seleccionadas para el modelo: {features}")
print(f"Variable objetivo: {target}")

categorical_features = ['HomeTeam', 'AwayTeam', 'Division']
numerical_features = ['match_dayofweek', 'match_month', 'match_year'] 

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features), 
        ('num', 'passthrough', numerical_features) 
    ])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClases de resultados codificadas: {label_encoder.classes_} -> {np.unique(y_encoded)}")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nForma de los datos de entrenamiento (X_train): {X_train.shape}")
print(f"Forma de los datos de prueba (X_test): {X_test.shape}")

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))]) 

print("\nEntrenando el modelo...")
model_pipeline.fit(X_train, y_train)
print(" Modelo entrenado exitosamente.")

print("\nEvaluando el modelo en los datos de prueba...")
y_pred_encoded = model_pipeline.predict(X_test)

y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test_original, y_pred_original, zero_division=0))

accuracy = accuracy_score(y_test_original, y_pred_original)
print(f"Precisión (Accuracy): {accuracy:.4f}")

print("\n--- Matriz de Confusión ---")
conf_matrix = confusion_matrix(y_test_original, y_pred_original, labels=label_encoder.classes_)
print(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

print("\n--- Predicción de un nuevo partido (Ejemplo Futuro) ---")

from datetime import datetime, timedelta
today = datetime.today()
next_saturday = today + timedelta( (5-today.weekday()) % 7 ) 
if next_saturday < today: 
    next_saturday += timedelta(days=7)

print(f"Fecha del partido de ejemplo: {next_saturday.strftime('%Y-%m-%d')}")

new_match_data_dict = {
    'HomeTeam': 'Real Madrid',   
    'AwayTeam': 'Barcelona',     
    'Division': 'SP1',           
    'match_dayofweek': next_saturday.weekday(), 
    'match_month': next_saturday.month,
    'match_year': next_saturday.year
}
new_match_df = pd.DataFrame([new_match_data_dict])

new_match_df = new_match_df[features] 

print("\nDatos del partido a predecir:")
print(new_match_df)

predicted_outcome_encoded = model_pipeline.predict(new_match_df)
predicted_outcome_proba = model_pipeline.predict_proba(new_match_df) 

predicted_outcome_original = label_encoder.inverse_transform(predicted_outcome_encoded)

print(f"\nEl modelo predice el resultado para este partido: {predicted_outcome_original[0]}")

print("\nProbabilidades de cada resultado:")
for i, class_label in enumerate(label_encoder.classes_):
    print(f"  {class_label}: {predicted_outcome_proba[0][i]:.4f}")
