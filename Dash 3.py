import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import joblib
import time


# Inicializar Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Cargar Datos
try:
    df = pd.read_excel("Datos limpiados1.xlsx")
    df["Precio M²"] = df["price"] / df["square_feet"]
    df = df[df["square_feet"] < df["square_feet"].quantile(0.9)]
    df = df[df["price"] < df["price"].quantile(0.9)]
    df = df[df["Precio M²"] < df["Precio M²"].quantile(0.85)]
    
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
    "ND": "Midwest", "SD": "Midwest"})
    df["size_category"] = pd.cut(df["square_feet"], bins=[0, 700, 1200, 2455], labels=["Pequeño", "Mediano", "Grande"], include_lowest=True)

except FileNotFoundError:
    print("Error: Archivo 'Datos limpiados1.xlsx' no encontrado.")
    df = pd.DataFrame()

#Cargar modelo oferta
best_rf_oferta = joblib.load("modelo_random_forest.pkl")
label_encoder = joblib.load("label_encoder.pkl")

#Cargar modelo precio
best_rf_precio = joblib.load("xgboost1.pkl")

# Calcular rentabilidad del alquiler
if not df.empty and "price" in df.columns and "Precio M²" in df.columns:
    df["rental_yield"] = (df["Precio M²"] * 12) / df["price"] * 100

# Función para predecir el precio
def predict_price(state, square_feet, pool, dishwasher, parking, refrigerator, pets_allowed, bathrooms):
    input_data = pd.DataFrame({
        "square_feet": [square_feet],
        "Pool": [pool],
        "Dishwasher": [dishwasher],
        "Parking": [parking],
        "Refrigerator": [refrigerator],
        "pets_allowed": [pets_allowed],
        "bathrooms": [bathrooms],
        "time": [int(time.time())]
    })

    # Codificar `state`
    for s in best_rf_precio.feature_names_in_:
        if s.startswith("state_"):
            input_data[s] = 1 if s == f"state_{state}" else 0

    # Asegurar columnas correctas
    missing_cols = set(best_rf_precio.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    input_data = input_data[best_rf_precio.feature_names_in_]

    return best_rf_precio.predict(input_data)[0]

# Función para predecir la oferta
def predict_rent(bedrooms, bathrooms, price_per_sqft, region):
    valid_regions = ['region_Midwest', 'region_South', 'region_West']
    region_encoded = {r: 0 for r in valid_regions}
    if region in valid_regions:
        region_encoded[region] = 1

    input_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "price_per_sqft": [price_per_sqft],
        **region_encoded
    })
    region_df = df[df["region"] == region]
    total_apartments_region = len(region_df)
    category_count_region = len(region_df[region_df["size_category"] == label_encoder.inverse_transform([best_rf_oferta.predict(input_data)[0]])[0]])
    return f"Tamaño Estimado: {label_encoder.inverse_transform([best_rf_oferta.predict(input_data)[0]])[0]}, Oferta en la region {region}: {round(category_count_region/total_apartments_region*100,2)}%"

# Layout
app.layout = html.Div([
    html.H1("Análisis de Mercado Inmobiliario", style={'textAlign': 'center', 'color': 'purple'}),
    
    html.H3("Zonas más Rentables para Invertir", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    dcc.Graph(id='map-graph', figure=px.scatter_map(df, lat="latitude", lon="longitude", color="Precio M²", zoom=3, title="Precio de Alquiler por M²", range_color=[df["Precio M²"].min(), df["Precio M²"].max()])),

    html.H3("Predicción de Clasificación de Oferta", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Número de Habitaciones:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Input(id='bedrooms-input', type='number', value=1),
    html.Label("Número de Baños:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Input(id='bathrooms-input', type='number', value=1),
    html.Label("Precio por pies cuadrados:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Input(id='price-per-sqft-input', type='number', value=50),
    html.Label("Región:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='region-dropdown', options=[{'label': i, 'value': i} for i in df["region"].dropna().unique()], value="West"),
    html.Div(id='rent-prediction-output', style={'marginTop': 20, 'fontSize': 22, 'fontWeight': 'bold', 'color': 'indigo', 'textAlign': 'center'}),
    
    html.H3("Análisis de Precios de Propiedades Similares", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Número de Habitaciones",style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='bed-filter', options=[{'label': i, 'value': i} for i in df["bedrooms"].dropna().unique()], multi=True),
    html.Label("Número de Baños", style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='bath-filter', options=[{'label': i, 'value': i} for i in df["bathrooms"].dropna().unique()], multi=True),
    html.Label("Rango de Precios", style={'textAlign': 'left','color':'gray'}),
    dcc.RangeSlider(id='price-range', min=df["price"].min(), max=df["price"].max(), step=150, value=[df["price"].min(), df["price"].max()]),
    dcc.Graph(id='histogram'),
    dcc.Graph(id='boxplot'),
    
    html.H3("Predicción de Precio Optimo Alquiler", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Estado:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='state-dropdown', options=[{'label': i, 'value': i} for i in df["state"].dropna().unique()], value="MD"),
    html.Label("Área en pies cuadrados:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Input(id='square-feet-input', type='number', value=1),
    html.Label("Piscina:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='pool-dropdown', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=0),
    html.Label("Lavaplatos:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='dishwasher-dropdown', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=0),
    html.Label("Parqueadero:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='parking-dropdown', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=0),
    html.Label("Refrigerador:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='refrigerator-dropdown', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=0),
    html.Label("Mascotas Permitidas:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='pets-dropdown', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=0),
    html.Label("Número de Baños:", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Input(id='bathrooms-input-2', type='number', value=1),
    html.Div(id='price-prediction-output', style={'marginTop': 20, 'fontSize': 22, 'fontWeight': 'bold', 'color': 'indigo', 'textAlign': 'center'}),

    html.H3("Factores que más Impactan en el Precio", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    dcc.Graph(id='correlation-matrix'),
    dcc.Graph(id='scatter-plots'),
    dcc.Input(id='update-trigger', value='', type='text', style={'display': 'none'}),
    html.Div(id='feature-importance-output', style={'marginTop': 20, 'fontSize': 22, 'fontWeight': 'bold', 'color': 'indigo', 'textAlign': 'center'}),

    html.H3("Comparación de Precios entre Ciudades", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Selecciona un Estado",style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='state-dropdown-2', options=[{'label': i, 'value': i} for i in df["state"].dropna().unique()], value=None),
    dcc.Graph(id='city-price-boxplot'),
    dcc.Graph(id='city-price-histogram'),

    html.H3("Rentabilidad de la Inversión en Vivienda", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Selecciona un Estado", style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='state-yield-dropdown', options=[{'label': i, 'value': i} for i in df["state"].dropna().unique()], value=None),
    dcc.Graph(id='yield-bar-chart'),
    dcc.Graph(id='yield-heatmap'),

    ])

# Callbacks
@app.callback(
    Output('price-prediction-output', 'children'),
    [Input('state-dropdown', 'value'), Input('square-feet-input', 'value'), Input('pool-dropdown', 'value'),
     Input('dishwasher-dropdown', 'value'), Input('parking-dropdown', 'value'), Input('refrigerator-dropdown', 'value'),
     Input('pets-dropdown', 'value'), Input('bathrooms-input-2', 'value')]
)
def update_price_prediction(*args):
    return f"Precio estimado: ${predict_price(*args):.2f}"

@app.callback(
    Output('rent-prediction-output', 'children'),
    [Input('bedrooms-input', 'value'), Input('bathrooms-input', 'value'), Input('price-per-sqft-input', 'value'), 
     Input('region-dropdown', 'value')]
)
def update_rent_prediction(*args):
    return f"Clasificación: {predict_rent(*args)}"

@app.callback(
    [Output('histogram', 'figure'), Output('boxplot', 'figure')],
    [Input('bed-filter', 'value'), Input('bath-filter', 'value'), Input('price-range', 'value')]
)
def update_price_visuals(beds, baths, price_range):
    filtered_df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
    if beds:
        filtered_df = filtered_df[filtered_df["bedrooms"].isin(beds)]
    if baths:
        filtered_df = filtered_df[filtered_df["bathrooms"].isin(baths)]
    
    hist_fig = px.histogram(filtered_df, x="price", title="Distribución de Precios de Propiedades Similares",  labels={"price": "Precio de la Propiedad (USD)", "count": "Número de Propiedades"})
    hist_fig.update_layout(
        xaxis_title="Precio de la Propiedad (USD)",
        yaxis_title="Número de Propiedades")

    box_fig = px.box(filtered_df, y="price", title="Boxplot de Precios de Propiedades Similares", labels={"price": "Precio de la Propiedad (USD)"})
    return hist_fig, box_fig

# Callback para actualizar matriz de correlación y scatter plots
@app.callback(
    [Output('correlation-matrix', 'figure'), Output('scatter-plots', 'figure')],
    Input('update-trigger', 'value')
)
def update_correlation_and_scatter(_):
    if df.empty or len(df.columns) < 2:
        return px.imshow([]), px.scatter()
    
    rename_dict = {
        "id":"Nr. Identificación",
        "price": "Precio (USD)",
        "bedrooms": "Habitaciones",
        "bathrooms": "Baños",
        "square_feet": "Metros Cuadrados",
        "latitude":"Latitud",
        "longitude":"Longitud",
        "time":"Tiempo",
        "Refrigerator": "Refrigerador",
        "Pool": "Piscina",
        "Dishwasher":"Lavaplatos",
        "Parking":"Parqueadero",
        "Precio M²":"Precio M²",
        "rental_yield":"Rentabilidad Inversion"
    }

    # Renombrar columnas temporalmente para las gráficas
    df_renamed = df.rename(columns=rename_dict)
    
    correlation_matrix =  df_renamed.corr(numeric_only=True)
    fig_corr = px.imshow(correlation_matrix, title="Matriz de Correlación", color_continuous_scale='viridis',labels=dict(x="Parámetro", y="Parámetro", color="Correlación"),
        x=correlation_matrix.columns,y=correlation_matrix.columns)
    
    scatter_fig = px.scatter_matrix(df, dimensions=["price", "bedrooms", "bathrooms", "square_feet"], labels={"price":"Precio (USD)", "bedrooms":"Habitaciones", "bathrooms":"Baños", "square_feet":"Metros Cuadrados"},color="price", title="Relación entre Precio y Factores")
    
    return fig_corr, scatter_fig

@app.callback(
    [Output('city-price-boxplot', 'figure'),
     Output('city-price-histogram', 'figure')],
    Input('state-dropdown-2', 'value')
)
def update_city_price_visuals(selected_state):
    if df.empty or 'state' not in df.columns or 'cityname' not in df.columns or 'price' not in df.columns:
        return px.box(title="No hay datos disponibles"), px.histogram(title="No hay datos disponibles")

    # Filtrar por estado seleccionado
    filtered_df = df[df['state'] == selected_state] if selected_state else df

    # Gráfico de boxplot (Distribución de precios por ciudad)
    boxplot_fig = px.box(filtered_df, x='cityname', y='price', title=f"Distribución de Precios en {selected_state}", points="all", labels={"cityname": "Ciudad", "price": "Precio (USD)"})

    # Gráfico de histograma (Frecuencia de precios)
    hist_fig = px.histogram(filtered_df, x='price', title=f"Distribución de Precios de Propiedades en {selected_state}", nbins=30, labels={"price": "Precio (USD)", "count": "Frecuencia"})
    hist_fig.update_layout(
    xaxis_title="Precio (USD)",
    yaxis_title="Frecuencia"
)

    return boxplot_fig, hist_fig

@app.callback(
    [Output('yield-bar-chart', 'figure'),
     Output('yield-heatmap', 'figure')],
    Input('state-yield-dropdown', 'value')
)
def update_yield_analysis(selected_state):
    if df.empty or 'state' not in df.columns or 'cityname' not in df.columns or 'rental_yield' not in df.columns:
        return px.bar(title="No hay datos disponibles"), px.density_mapbox(title="No hay datos disponibles")

    # Filtrar por estado seleccionado
    filtered_df = df[df['state'] == selected_state] if selected_state else df

    # Gráfico de barras de rentabilidad por ciudad
    bar_fig = px.bar(filtered_df, x="cityname", y="rental_yield", title=f"Rentabilidad del Alquiler por Ciudad en {selected_state}", labels={"rental_yield": "Rentabilidad (%)", "cityname":"Ciudad"}, color="rental_yield", color_continuous_scale="viridis")

    # Mapa de calor de rentabilidad
    heatmap_fig = px.density_map(filtered_df, lat="latitude", lon="longitude", z="rental_yield",
                             radius=10, title=f"Mapa de Rentabilidad del Alquiler en {selected_state}",
                             color_continuous_scale="plasma", labels={"rental_yield":"Rentabilidad (%)"}, zoom=3)
 

    return bar_fig, heatmap_fig


if __name__ == '__main__':
    app.run_server(debug=True)
