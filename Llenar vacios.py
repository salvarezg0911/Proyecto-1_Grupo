# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:20:05 2025

@author: Stefania Alvarez
"""

import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

#Importacion de datos
archivo = "Datos Iniciales.xlsx"

hoja = pd.read_excel(io = archivo)

#-----------------------------
#    1. Category
#-----------------------------
#Eliminar las 4 filas que en la variable category correspondian a short-term y home
hoja = hoja[hoja["category"] == "housing/rent/apartment"]

#-----------------------------
#    2. Ttile y body
#-----------------------------
hoja = hoja.drop(columns=["title",'body'])

#-----------------------------
#    3. Amenities
#-----------------------------

amenities = ["Refrigerator", "Pool", "Dishwasher" , "Parking"]

for j in amenities:
    hoja[j] = 0

    for a,i in hoja['amenities'].items():
        if pd.isna(i):
            continue
        if j in i.split(','):
            hoja.at[a,j] =1 

hoja = hoja.drop(columns=["amenities"])      
    

#-----------------------------
#    4.Bathrooms
#-----------------------------

ID = [i for i in hoja['id']]


for i in hoja.index:
    porcentaje_min = 0
    porcentaje_max = 0
    mediana_b =0 
    total_b = []
    
    if pd.isna(hoja.at[i,'bathrooms']):
        porcentaje_min = hoja.at[i, 'square_feet']*(1-0.1)
        porcentaje_max = hoja.at[i, 'square_feet']*(1+0.1)
        
        for j in hoja.index:
            if hoja.at[j, 'square_feet'] <= porcentaje_max and hoja.at[j, 'square_feet'] >= porcentaje_min and pd.notna(hoja.at[j, 'bathrooms']):
                total_b.append(hoja.at[j, 'bathrooms'])
                
        if len(total_b)!= 0:
            mediana_b = st.median(total_b)
            hoja.at[i, 'bathrooms'] = mediana_b
        else:
            hoja.at[i, 'bathrooms'] = 1
     
        
#-----------------------------
#    5. Bedrooms y State
#-----------------------------
cod_estados = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
}
hoja.dropna(subset=['latitude'], inplace=True)

#Funcion busqueda de estado y ciudad
def get_location(lat, lon):
    geolocator = Nominatim(user_agent="geo_finder")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location:
            address = location.raw.get("address", {})
            city = city = address.get("city", address.get("town", address.get( "village", address.get("hamlet", address.get("suburb", address.get("locality", address.get("county", "Unknown"))))))) 
            statein = address.get("state", "Unknown")
            state = cod_estados.get(statein)
            return city, state
        return "Unknown", "Unknown"
    except GeocoderTimedOut:
        return "Timeout", "Timeout"

for i in hoja.index:
    porcentaje_min = 0
    porcentaje_max = 0
    mediana_b =0 
    total_b = []
    
    if pd.isna(hoja.at[i,'bedrooms']):
        porcentaje_min = hoja.at[i, 'square_feet']*(1-0.1)
        porcentaje_max = hoja.at[i, 'square_feet']*(1+0.1)
        
        for j in hoja.index:
            if hoja.at[j, 'square_feet'] <= porcentaje_max and hoja.at[j, 'square_feet'] >= porcentaje_min and pd.notna(hoja.at[j, 'bedrooms']):
                total_b.append(hoja.at[j,'bedrooms'])
                
        if len(total_b)!= 0:
            mediana_b = st.median(total_b)
            hoja.at[i,'bedrooms'] = mediana_b
        else:
            hoja.at[i,'bedrooms'] = 1
    if pd.isna(hoja.at[i,'state']):
        lat=hoja.at[i, 'latitude']
        lon=hoja.at[i, 'longitude']
        ciudad, estado=get_location(lat, lon)
        hoja.at[i, 'state']=estado
        hoja.at[i, 'cityname']=ciudad




#-----------------------------
#    6. Currency y fee
#-----------------------------
hoja = hoja.drop(columns=["currency",'fee'])

#-----------------------------
#    7. has_photo
#-----------------------------
hoja = hoja.drop(columns=["has_photo"])

#-----------------------------
#    8. Pets allowed
#-----------------------------


for i in hoja.index:  
    if pd.isna(hoja.at[i,'pets_allowed']):
        hoja.at[i,'pets_allowed'] = "None"


#-----------------------------
#    9. price y square_feet
#-----------------------------
#NO HAY CAMBIOS

#-----------------------------
#    10. price_display, price_type y address
#-----------------------------
hoja = hoja.drop(columns=["price_display","price_type","address"]) 

 #-----------------------------
 #    11. source y time
 #----------------------------- 
hoja = hoja.drop(columns=["source"]) 


#-----------------------------
# Exportar al data frame
#----------------------------- 
hoja.to_excel("Datos limpiados1.xlsx", index=False)
