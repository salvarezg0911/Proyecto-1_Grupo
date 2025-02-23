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


# Inicializar Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Cargar Datos
try:
    df = pd.read_excel("Datos limpiados1.xlsx")
except FileNotFoundError:
    print("Error: Archivo 'Datos limpiados1.xlsx' no encontrado.")
    df = pd.DataFrame()

# Crear columna de precio por metro cuadrado si existen los datos necesarios
if not df.empty and "price" in df.columns and "square_feet" in df.columns:
    df["Precio M²"] = df["price"] / df["square_feet"]
# Calcular rentabilidad del alquiler
if not df.empty and "price" in df.columns and "Precio M²" in df.columns:
    df["rental_yield"] = (df["Precio M²"] * 12) / df["price"] * 100

# Variables para los modelos
features = ["cityname", "bedrooms", "bathrooms"]
X_price = df[features].dropna()
y_price = df.loc[X_price.index, "price"].dropna()
X_rent = df[features].dropna()
y_rent = df.loc[X_rent.index, "Precio M²"].dropna()

model_price, model_rent = None, None
if not X_price.empty and len(X_price) == len(y_price):
    model_price = LinearRegression()
    X_encoded_price = pd.get_dummies(X_price, columns=["cityname"], drop_first=True)
    model_price.fit(X_encoded_price, y_price)

if not X_rent.empty and len(X_rent) == len(y_rent):
    model_rent = LinearRegression()
    X_encoded_rent = pd.get_dummies(X_rent, columns=["cityname"], drop_first=True)
    model_rent.fit(X_encoded_rent, y_rent)

# Filtrar datos válidos para series temporales
if not df.empty and "date" in df.columns and "price" in df.columns and "cityname" in df.columns:
    df_time_series = df.groupby(["date", "cityname"]).agg({"price": "mean"}).reset_index()
else:
    df_time_series = pd.DataFrame()

# Funciones de predicción
def predict_price(city, bedrooms, bathrooms):
    if model_price:
        input_data = pd.DataFrame([[city, bedrooms, bathrooms]], columns=features)
        input_encoded = pd.get_dummies(input_data, columns=["cityname"], drop_first=True)
        input_encoded = input_encoded.reindex(columns=X_encoded_price.columns, fill_value=0)
        return model_price.predict(input_encoded)[0]
    return "Datos insuficientes"

def predict_rent(city, bedrooms, bathrooms):
    if model_rent:
        input_data = pd.DataFrame([[city, bedrooms, bathrooms]], columns=features)
        input_encoded = pd.get_dummies(input_data, columns=["cityname"], drop_first=True)
        input_encoded = input_encoded.reindex(columns=X_encoded_rent.columns, fill_value=0)
        return model_rent.predict(input_encoded)[0]
    return "Datos insuficientes"

# Layout
app.layout = html.Div([
    html.H1("Análisis de Mercado Inmobiliario", style={'textAlign': 'center', 'color': 'purple'}),
    
    html.H3("1. Zonas más Rentables para Invertir", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    dcc.Graph(id='map-graph', figure=px.scatter_map(df, lat="latitude", lon="longitude", color="Precio M²", zoom=3, title="Precio de Alquiler por M²")),
    #
    html.H3("       Predicción de Rentabilidad", style={'textAlign': 'left','color':'gray'}),
    html.Label("Ciudad", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='city-dropdown', options=[{'label': i, 'value': i} for i in df["cityname"].dropna().unique()], value=None),
    html.Label("Número de Habitaciones", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='bed-dropdown', options=[{'label': i, 'value': i} for i in df["bedrooms"].dropna().unique()], value=None),
    html.Label("Número de Baños", style={'textAlign': 'left', 'color': 'gray'}),
    dcc.Dropdown(id='bath-dropdown', options=[{'label': i, 'value': i} for i in df["bathrooms"].dropna().unique()], value=None),
    html.Div(id='rent-prediction-output', style={'marginTop': 20, 'fontSize': 22, 'fontWeight': 'bold', 'color': 'darkslateblue', 'textAlign': 'center'}),
    
    html.H3("2. Análisis de Precios de Propiedades Similares", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Número de Habitaciones",style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='bed-filter', options=[{'label': i, 'value': i} for i in df["bedrooms"].dropna().unique()], multi=True),
    html.Label("Número de Baños", style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='bath-filter', options=[{'label': i, 'value': i} for i in df["bathrooms"].dropna().unique()], multi=True),
    html.Label("Rango de Precios", style={'textAlign': 'left','color':'gray'}),
    dcc.RangeSlider(id='price-range', min=df["price"].min(), max=df["price"].max(), step=10000, value=[df["price"].min(), df["price"].max()]),
    dcc.Graph(id='histogram'),
    dcc.Graph(id='boxplot'),
    #
    html.H3("      Predicción de Precio Óptimo para Venta o Alquiler", style={'textAlign': 'left','color':'gray'}),
    html.Div(id='price-prediction-output', style={'marginTop': 20, 'fontSize': 22, 'fontWeight': 'bold', 'color': 'darkslateblue', 'textAlign': 'center'}),

    html.H3("3. Factores que más Impactan en el Precio", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    dcc.Graph(id='correlation-matrix'),
    dcc.Graph(id='scatter-plots'),
    dcc.Input(id='update-trigger', value='', type='text', style={'display': 'none'}),
    html.Div(id='feature-importance-output', style={'marginTop': 20, 'fontSize': 22, 'fontWeight': 'bold', 'color': 'darkblue', 'textAlign': 'center'}),

    html.H3("4. Comparación de Precios entre Ciudades", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Selecciona un Estado",style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='state-dropdown', options=[{'label': i, 'value': i} for i in df["state"].dropna().unique()], value=None),
    dcc.Graph(id='city-price-boxplot'),
    dcc.Graph(id='city-price-histogram'),

    html.H3("5. Rentabilidad de la Inversión en Vivienda", style={'textAlign': 'left', 'color': 'rebeccapurple'}),
    html.Label("Selecciona un Estado", style={'textAlign': 'left','color':'gray'}),
    dcc.Dropdown(id='state-yield-dropdown', options=[{'label': i, 'value': i} for i in df["state"].dropna().unique()], value=None),
    dcc.Graph(id='yield-bar-chart'),
    dcc.Graph(id='yield-heatmap'),

    ])

# Callbacks
@app.callback(
    Output('rent-prediction-output', 'children'),
    [Input('city-dropdown', 'value'), Input('bed-dropdown', 'value'), Input('bath-dropdown', 'value')]
)
def update_rent_prediction(city, bedrooms, bathrooms):
    if city and bedrooms and bathrooms:
        rent_prediction = predict_rent(city, bedrooms, bathrooms)
        return f"Rentabilidad estimada por m²: ${rent_prediction:.2f}"
    return "Seleccione todos los valores para obtener la predicción."

@app.callback(
    Output('price-prediction-output', 'children'),
    [Input('city-dropdown', 'value'), Input('bed-dropdown', 'value'), Input('bath-dropdown', 'value')]
)
def update_price_prediction(city, bedrooms, bathrooms):
    if city and bedrooms and bathrooms:
        price_prediction = predict_price(city, bedrooms, bathrooms)
        return f"Precio estimado: ${price_prediction:.2f}"
    return "Seleccione todos los valores para obtener la predicción."

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
    
    hist_fig = px.histogram(filtered_df, x="price", title="Distribución de Precios de Propiedades Similares")
    box_fig = px.box(filtered_df, y="price", title="Boxplot de Precios de Propiedades Similares")
    return hist_fig, box_fig

# Callback para actualizar matriz de correlación y scatter plots
@app.callback(
    [Output('correlation-matrix', 'figure'), Output('scatter-plots', 'figure')],
    Input('update-trigger', 'value')
)
def update_correlation_and_scatter(_):
    if df.empty or len(df.columns) < 2:
        return px.imshow([]), px.scatter()
    
    correlation_matrix = df.corr(numeric_only=True)
    fig_corr = px.imshow(correlation_matrix, title="Matriz de Correlación", color_continuous_scale='viridis')
    
    scatter_fig = px.scatter_matrix(df, dimensions=["price", "bedrooms", "bathrooms", "square_feet"], color="price", title="Relación entre Precio y Factores")
    
    return fig_corr, scatter_fig

# Callback para calcular la importancia de variables en el precio
@app.callback(
    Output('feature-importance-output', 'children'),
    Input('update-trigger', 'value')
)
def update_feature_importance(_):
    if model_price:
        importance = pd.Series(model_price.coef_, index=X_encoded_price.columns).sort_values(ascending=False)
        importance_text = "\n".join([f"{var}: {imp:.2f}" for var, imp in importance.items()])
        #return f"Factores que más impactan en el precio:\n{importance_text}"
    #return "No hay suficientes datos para calcular la importancia de las variables."

@app.callback(
    [Output('city-price-boxplot', 'figure'),
     Output('city-price-histogram', 'figure')],
    Input('state-dropdown', 'value')
)
def update_city_price_visuals(selected_state):
    if df.empty or 'state' not in df.columns or 'cityname' not in df.columns or 'price' not in df.columns:
        return px.box(title="No hay datos disponibles"), px.histogram(title="No hay datos disponibles")

    # Filtrar por estado seleccionado
    filtered_df = df[df['state'] == selected_state] if selected_state else df

    # Gráfico de boxplot (Distribución de precios por ciudad)
    boxplot_fig = px.box(filtered_df, x='cityname', y='price', title=f"Distribución de Precios en {selected_state}", points="all")

    # Gráfico de histograma (Frecuencia de precios)
    hist_fig = px.histogram(filtered_df, x='price', title=f"Distribución de Precios de Propiedades en {selected_state}", nbins=30)

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
    bar_fig = px.bar(filtered_df, x="cityname", y="rental_yield", title=f"Rentabilidad del Alquiler por Ciudad en {selected_state}", labels={"rental_yield": "Rentabilidad (%)"}, color="rental_yield", color_continuous_scale="viridis")

    # Mapa de calor de rentabilidad
    heatmap_fig = px.density_map(filtered_df, lat="latitude", lon="longitude", z="rental_yield",
                             radius=10, title=f"Mapa de Rentabilidad del Alquiler en {selected_state}",
                             color_continuous_scale="plasma")
    return bar_fig, heatmap_fig


if __name__ == '__main__':
    app.run_server(debug=True)
