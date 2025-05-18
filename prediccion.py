import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Un buen modelo de clasificación para empezar
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# === CARGAR LOS DATOS ===
input_file = r"C:\Users\LENOVO\Downloads\Club-Football-Match-Data-2000-2025-main\Club-Football-Match-Data-2000-2025-main\data\Matches.csv"

try:
    df = pd.read_csv(input_file, low_memory=False, encoding='utf-8')
    print(" Datos CSV cargados exitosamente.")
except Exception as e:
    print(f" Error al cargar el CSV: {e}")
    print("Asegúrate de que la ruta del archivo sea correcta y la codificación.")
    exit() 

ligas_objetivo = [
    'SP1', 'D1',
    'E0', 'I1', 'F1'
]
df_filtrado = df[df['Division'].isin(ligas_objetivo)].copy()

def calcular_resultado(row):
    home = row['FTHome']
    away = row['FTAway']
    if pd.isna(home) or pd.isna(away):
        return "Unknown" 
    if home > away:
        return "Win"
    elif home < away:
        return "Loss"
    else:
        return "Draw"

df_filtrado['match_result'] = df_filtrado.apply(calcular_resultado, axis=1)

df_filtrado = df_filtrado[df_filtrado['match_result'] != 'Unknown'].copy()

df_filtrado['MatchDate'] = pd.to_datetime(df_filtrado['MatchDate'], errors='coerce')
df_filtrado = df_filtrado.dropna(subset=['MatchDate']) 

df_filtrado['match_dayofweek'] = df_filtrado['MatchDate'].dt.dayofweek 
df_filtrado['match_month'] = df_filtrado['MatchDate'].dt.month
df_filtrado['match_year'] = df_filtrado['MatchDate'].dt.year

df_filtrado = df_filtrado.dropna(subset=['HomeTeam', 'AwayTeam', 'Division', 'FTHome', 'FTAway', 'match_result']).copy()

df_filtrado = df_filtrado.drop_duplicates().copy()

print(f"Datos preprocesados. Total de partidos válidos para ML: {len(df_filtrado)}")
print("Distribución de resultados:")
print(df_filtrado['match_result'].value_counts())

# === SELECCIONAR CARACTERÍSTICAS Y VARIABLE OBJETIVO ===
features = ['HomeTeam', 'AwayTeam', 'Division', 'FTHome', 'FTAway', 'match_dayofweek', 'match_month', 'match_year']
target = 'match_result'

X = df_filtrado[features]
y = df_filtrado[target]

print(f"\nCaracterísticas seleccionadas: {features}")
print(f"Variable objetivo: {target}")

# === PREPROCESAMIENTO DE DATOS ===
categorical_features = ['HomeTeam', 'AwayTeam', 'Division']
numerical_features = ['FTHome', 'FTAway', 'match_dayofweek', 'match_month', 'match_year']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features) # 'passthrough' para no transformar numéricas o 'scaler', StandardScaler()
    ])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClases de resultados codificadas: {label_encoder.classes_}")

# === DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nForma de los datos de entrenamiento (X_train): {X_train.shape}")
print(f"Forma de los datos de prueba (X_test): {X_test.shape}")

# === ENTRENAR UN MODELO DE CLASIFICACIÓN ===
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

print("\nEntrenando el modelo...")
model_pipeline.fit(X_train, y_train)
print(" Modelo entrenado exitosamente.")

# === EVALUAR EL MODELO ===
print("\nEvaluando el modelo en los datos de prueba...")
y_pred_encoded = model_pipeline.predict(X_test)

y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test_original, y_pred_original))

accuracy = accuracy_score(y_test_original, y_pred_original)
print(f"Precisión (Accuracy): {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test_original, y_pred_original, labels=label_encoder.classes_)
print("\n--- Matriz de Confusión ---")
print(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

# === PRUEBA CON UN NUEVO PARTIDO ===
print("\n--- Predicción de un nuevo partido (Ejemplo) ---")
new_match_data = pd.DataFrame([{
    'HomeTeam': 'Real Madrid',
    'AwayTeam': 'Barcelona',
    'Division': 'SP1',
    'FTHome': 2, 
    'FTAway': 1, # 
    'match_dayofweek': 6, 
    'match_month': 5, 
    'match_year': 2024
}])

new_match_data = new_match_data[features]

predicted_outcome_encoded = model_pipeline.predict(new_match_data)
predicted_outcome_original = label_encoder.inverse_transform(predicted_outcome_encoded)

print(f"El modelo predice el resultado para este partido hipotético: {predicted_outcome_original[0]}")