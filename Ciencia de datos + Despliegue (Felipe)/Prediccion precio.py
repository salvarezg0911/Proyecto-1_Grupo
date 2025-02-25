import pandas as pd
import joblib

#Cargar el modelo y el codificador de etiquetas
best_rf_model = joblib.load("Ciencia de datos + Despliegue (Felipe)/xgboost1.pkl")
label_encoder = joblib.load("Ciencia de datos + Despliegue (Felipe)/respuesta.pkl")
columnas_entrenamiento = best_rf_model.feature_names_in_
print(columnas_entrenamiento)

#Definir un nuevo apartamento para predecir
parametros = pd.DataFrame({
    "square_feet": [300],
    "Pool": [1],
    "Dishwasher": [0],
    "Parking": [0],
    "Refrigerator": [1],
    "bathrooms": [1],
    "time": [1568781099],
    "state_AL": [0],
    "state_AR": [0],
    "state_AZ": [0],
    "state_CA": [1],
    "state_CO": [0],
    "state_CT": [0],
    "state_DC": [0],
    "state_DE": [0],
    "state_FL": [0],
    "state_GA": [0],
    "state_HI": [0],
    "state_IA": [0],
    "state_ID": [0],
    "state_IL": [0],
    "state_IN": [0],
    "state_KS": [0],
    "state_KY": [0],
    "state_LA": [0],
    "state_MA": [0],
    "state_MD": [0],
    "state_ME": [0],
    "state_MI": [0],
    "state_MN": [0],
    "state_MO": [0],
    "state_MS": [0],
    "state_MT": [0],
    "state_NC": [0],
    "state_ND": [0],
    "state_NE": [0],
    "state_NH": [0],
    "state_NJ": [0],
    "state_NM": [0],
    "state_NV": [0],
    "state_NY": [0],
    "state_OH": [0],
    "state_OK": [0],
    "state_OR": [0],
    "state_PA": [0],
    "state_RI": [0],
    "state_SC": [0],
    "state_SD": [0],
    "state_TN": [0],
    "state_TX": [0],
    "state_UT": [0],
    "state_VA": [0],
    "state_VT": [0],
    "state_WA": [0],
    "state_WI": [0],
    "state_WV": [0],
    "state_WY": [0],
    "pets_allowed_Cats,Dogs": [0],
    "pets_allowed_Dogs": [1]
})


#Hacer la predicción
prediction = best_rf_model.predict(parametros)

print(prediction)

