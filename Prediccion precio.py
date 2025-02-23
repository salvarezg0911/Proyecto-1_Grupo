import pandas as pd
import joblib

# 📌 Cargar el modelo y el codificador de etiquetas
best_rf_model = joblib.load("xgboost1.pkl")
label_encoder = joblib.load("respuesta.pkl")
columnas_entrenamiento = best_rf_model.feature_names_in_
print(columnas_entrenamiento)

# 📌 Definir un nuevo apartamento para predecir
parametros = pd.DataFrame({
    "square_feet": [300],
    "price_per_sqft": [7.33],
    "Pool": [1],
    "Dishwasher": [0],
    "Parking": [0],
    "Refrigerator": [1],
    "bathrooms": [1],
    "region_Midwest": [1],
    "region_South": [1],
    "region_West": [0],
    "pets_allowed_Cats,Dogs": [0],
    "pets_allowed_Dogs": [1]
})

# 📌 Hacer la predicción
prediction = best_rf_model.predict(parametros)

print(prediction)

