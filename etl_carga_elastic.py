import pandas as pd
import json
from elasticsearch import Elasticsearch, helpers

# === CONFIGURACIÓN DE ARCHIVOS ===
input_file = r"C:\Users\LENOVO\Downloads\Club-Football-Match-Data-2000-2025-main\Club-Football-Match-Data-2000-2025-main\data\Matches.csv"
output_file = "matches_cleaned.json"
index_name = "football-matches"

# === LECTURA Y LIMPIEZA DE DATOS ===
df = pd.read_csv(input_file, low_memory=False, encoding='utf-8')

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

df_filtrado = df_filtrado.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHome', 'FTAway', 'MatchDate'])
df_filtrado = df_filtrado.drop_duplicates()

df_filtrado['MatchDate'] = pd.to_datetime(df_filtrado['MatchDate'], errors='coerce')
df_filtrado = df_filtrado.dropna(subset=['MatchDate'])
df_filtrado['MatchDate'] = df_filtrado['MatchDate'].dt.strftime('%Y-%m-%d')

columnas_finales = ['MatchDate', 'Division', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway', 'match_result']
df_final = df_filtrado[columnas_finales]

df_final.to_json(output_file, orient='records', lines=True, force_ascii=False)
print(f" Archivo JSON exportado: {output_file}")

# === CONEXIÓN A ELASTICSEARCH ===
es = Elasticsearch(
    "https://localhost:9200", 
    basic_auth=("elastic", "GOfWF97bXdUhUN7MPuDq"), 
    verify_certs=False, 
    ssl_show_warn=False 
)

# === CREAR ÍNDICE CON MAPEADO ===
mapping = {
    "mappings": {
        "properties": {
            "MatchDate": {
                "type": "date",
                "format": "yyyy-MM-dd"
            },
            "match_result": {
                "type": "keyword"
            }
        }
    }
}

if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f" Índice '{index_name}' eliminado.")

es.indices.create(index=index_name, body=mapping)
print(f" Índice '{index_name}' creado con mapeo personalizado.")

# === CARGAR DATOS ===
with open(output_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    data = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f" Error en línea {i+1}: {e}")

actions = [
    {
        "_index": index_name,
        "_source": doc
    }
    for doc in data
]

helpers.bulk(es, actions)
print(" Datos cargados exitosamente en Elasticsearch.")