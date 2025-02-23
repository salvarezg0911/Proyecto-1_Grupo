import pandas as pd
import joblib

file_path = "Datos limpiados1.xlsx"  # Asegúrate de que la ruta sea correcta
df = pd.read_excel(file_path, sheet_name="Sheet1")

#Crear una nueva columna de regiones
df["region"] = df["state"].map({
    "CA": "West", "NV": "West", "WA": "West", "OR": "West", "AZ": "West", "ID": "West",
    "MT": "West", "WY": "West", "UT": "West", "CO": "West", "AK": "West", "HI": "West",
    "TX": "South", "FL": "South", "GA": "South", "NC": "South", "TN": "South",
    "SC": "South", "AL": "South", "MS": "South", "KY": "South", "LA": "South",
    "AR": "South", "OK": "South", "WV": "South", "DC": "South", "VA": "South",
    "NY": "East", "NJ": "East", "PA": "East", "MA": "East", "MD": "East",
    "CT": "East", "RI": "East", "DE": "East", "NH": "East", "VT": "East", "ME": "East",
    "IL": "Midwest", "OH": "Midwest", "MI": "Midwest", "IN": "Midwest", "WI": "Midwest",
    "MN": "Midwest", "IA": "Midwest", "MO": "Midwest", "KS": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest"
})

#Definir categorías de tamaño del apartamento (Pequeño, Mediano, Grande)
df["size_category"] = pd.cut(df["square_feet"], bins=[0, 700, 1200, 2455], labels=["Pequeño", "Mediano", "Grande"], include_lowest=True)

#Cargar el modelo y el codificador de etiquetas
best_rf_model = joblib.load("modelo_random_forest.pkl")
label_encoder = joblib.load("label_encoder.pkl")
columnas_entrenamiento = best_rf_model.feature_names_in_
print(columnas_entrenamiento)

#Definir un nuevo apartamento para predecir
nuevo_apartamento = pd.DataFrame({
    "bedrooms": [2],
    "bathrooms": [1],
    "price_per_sqft": [25],
    "region_Midwest": [0],
    "region_South": [0],
    "region_West": [1]
})

#Hacer la predicción
prediction = best_rf_model.predict(nuevo_apartamento)
predicted_category = label_encoder.inverse_transform(prediction)[0]

#Calculo adicional
region_df = df[df["region"] == "West"]
total_apartments_region = len(region_df)
category_count_region = len(region_df[region_df["size_category"] == predicted_category])

print(f"🏡 Tamaño Predicho: {predicted_category}, Oferta en la region west: {round(category_count_region/total_apartments_region,2)*100}%")
