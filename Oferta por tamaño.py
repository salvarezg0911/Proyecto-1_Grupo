# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 📌 Cargar la base de datos desde el archivo Excel
file_path = "Datos limpiados1.xlsx"  # Asegúrate de que la ruta sea correcta
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 📌 Crear una nueva columna de regiones
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

# 📌 Filtrar valores atípicos del tamaño del apartamento
df = df[df["square_feet"] < df["square_feet"].quantile(0.99)]


# 📌 Definir categorías de tamaño del apartamento (Pequeño, Mediano, Grande)
df["size_category"] = pd.cut(df["square_feet"], bins=[0, 700, 1200, 2455], labels=["Pequeño", "Mediano", "Grande"], include_lowest=True)

# 📌 Crear una nueva variable de precio por metro cuadrado
df["price_per_sqft"] = df["price"] / df["square_feet"]
df = df[df["square_feet"] < df["square_feet"].quantile(0.99)]
df = df[df["price"] < df["price"].quantile(0.99)]
df = df[df["price_per_sqft"] > df["price_per_sqft"].quantile(0.02)]

print(df["size_category"].value_counts())
print(df.groupby("size_category")["square_feet"].agg(["min", "max"]))

# 📌 Variables para el modelo
features = ["bedrooms", "bathrooms", "price_per_sqft", "region"]
target = "size_category"

# 📌 Crear un DataFrame con las variables seleccionadas
df_model = df[features + [target]].copy()


# 📌 Aplicar One-Hot Encoding a 'region'
df_model = pd.get_dummies(df_model, columns=["region"], drop_first=True)

# 📌 Separar variables independientes (X) y dependiente (y)
X = df_model.drop(columns=[target])
y = df_model[target]

# 📌 Codificar la variable objetivo a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 📌 Dividir en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 Definir modelo Random Forest con optimización de hiperparámetros
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [2, 4, 6]
}

# 📌 Aplicar GridSearchCV con validación cruzada (CV=5) para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# 📌 Mejor modelo encontrado
best_rf_model = grid_search.best_estimator_

# 📌 Aplicar validación cruzada adicional con 10 folds
cv_scores = cross_val_score(best_rf_model, X, y, cv=10, scoring="accuracy")

# 📌 Entrenar modelo final en los datos de entrenamiento
best_rf_model.fit(X_train, y_train)

# 📌 Predicciones
y_pred_rf = best_rf_model.predict(X_test)

# 📌 Evaluación del modelo
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"📌 Accuracy del modelo Random Forest en test: {accuracy_rf:.4f}")
print(f"📌 Accuracy Promedio en Cross-Validation (CV=10): {cv_scores.mean():.4f}")
print(f"📌 Desviación Estándar en CV: {cv_scores.std():.4f}")

# 📌 Reporte de clasificación
print("\n📌 Reporte de clasificación:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# 📌 Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Clasificación de Tamaño de Apartamentos")
plt.show()

# 📌 Guardar el modelo y el LabelEncoder
joblib.dump(grid_search.best_estimator_, "modelo_random_forest.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")