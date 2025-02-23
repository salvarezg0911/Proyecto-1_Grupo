# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar la base de datos
file_path = "Datos limpiados1.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Eliminar el 1% más alto de los precios y mestros cuadrados
num_filas = df.shape[0]
print(f"El DataFrame tiene {num_filas} filas.")
media = df["price"].mean()
m=df["price"].max()
print(media)
print(m)
desviacion = df["price"].std()
print(desviacion)
limite_inferior = media - 1.8 * desviacion
limite_superior = media + 1.8 * desviacion
df=df[(df["price"] > limite_inferior) & (df["price"] < limite_superior)]

media = df["square_feet"].mean()
print(media)
desviacion = df["square_feet"].std()
print(desviacion)
limite_inferior = media - 1.8 * desviacion
limite_superior = media + 1.8 * desviacion
df=df[(df["square_feet"] > limite_inferior) & (df["square_feet"] < limite_superior)]

# Seleccionar variables para el modelo
features = ["square_feet","state", "Pool", "Dishwasher", "Parking", "Refrigerator", "pets_allowed", "bathrooms", "time"]
target = "price"

# Crear una copia del DataFrame original
df_model = df[features + [target]].copy()

#Revisar que la base tenga al menos 9000 datos
num_filas = df_model.shape[0]
print(f"El DataFrame tiene {num_filas} filas.")

# Variables dummies (1,0)
df_model = pd.get_dummies(df_model, columns=["state"], drop_first=True)
df_model = pd.get_dummies(df_model, columns=["pets_allowed"], drop_first=True)

# Separar variables independientes (X) y dependiente (y)
X = df_model.drop(columns=[target])
y = df_model[target]

# Dividir en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Se corrieron los mejores parametros pero para corregir sobre ajuste se escogieron estos
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 3, 7],
    "gamma": [0.1, 0.3, 0.5]
}

grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Entrenar modelo con los mejores hiperparámetros encontrados
best_xgb_model =  grid_search.best_estimator_
best_xgb_model.fit(X_train, y_train)

# Predicciones con XGBoost
y_pred_xgb = best_xgb_model.predict(X_test)

# Evaluación del modelo XGBoost
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# Validación cruzada con 20 folds en XGBoost
cv_scores_xgb = cross_val_score(best_xgb_model, X, y, cv=20, scoring="r2")

# Obtener la importancia de las variables en XGBoost
importances_xgb = best_xgb_model.feature_importances_
feature_names = X.columns
importancia_df = pd.DataFrame({"Variable": feature_names, "Importancia": importances_xgb})
print(importancia_df.sort_values(by="Importancia", ascending=False))

# Mostrar métricas del modelo XGBoost y validación cruzada
for i in cv_scores_xgb:
    print(i)
resultados_xgb = {
    "Mejor MAE (XGBoost)": mae_xgb,
    "Mejor RMSE (XGBoost)": rmse_xgb,
    "Mejor R² (XGBoost)": r2_xgb,
    "R² Promedio (CV XGBoost)": np.mean(cv_scores_xgb),
    "Desviación estándar (CV XGBoost)": np.std(cv_scores_xgb)
}


print("Resultados del modelo XGBoost:")
for k, v in resultados_xgb.items():
    print(f"{k}: {v:.4f}")

# Graficar valores reales vs predichos
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="blue", label="Predicciones XGBoost")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Línea Perfecta (y_real = y_pred)")

# Etiquetas
plt.xlabel("Precio Real (USD)")
plt.ylabel("Precio Predicho (USD)")
plt.title("Comparación entre Precios Reales y Predichos (XGBoost)")
plt.legend()
plt.grid(True)

# Mostrar gráfico
plt.show()

# Guardar el modelo y el LabelEncoder
#joblib.dump(np.mean(df_model["price_per_sqft"]), "promedio.pkl")
joblib.dump(grid_search.best_estimator_, "xgboost1.pkl")
joblib.dump(y, "respuesta.pkl")