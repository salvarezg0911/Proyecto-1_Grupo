# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:02:18 2025

@author: Stefania Alvarez
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import seaborn as sns

file_name = r"C:\Users\Stefania Alvarez\Documents\Universidad\2025-1\Analitica\Proyecto 1\Copia de Datos_Iniciales.xlsx"

df = pd.read_excel(file_name)

print(df.head())

#-----------------------------
#    Category
#-----------------------------
category = [i for i in df['category'] if False == pd.isna(i)]
conteo_category = []

for i in category:
    conteo_category.append(i.split('/')[-1])

ste=0
a=0
h=0
for i in conteo_category:
    if i=="short_term":
       ste+=1
    elif i=="apartment":
        a+=1
    else:
        h+=1   

#-----------------------------
#    Amenities
#-----------------------------
amenities_cat = df['amenities'].str.split(',').explode().str.strip().drop_duplicates().tolist()
amenities_dic = {}
amenities_dic["Null"] = df['amenities'].isnull().sum()

amenities = [i for i in df['amenities'] if False == pd.isna(i)]

for j in amenities_cat[1:]:
    count =0
    for i in amenities:
        if j in i.split(','):
            count+=1
    amenities_dic[j] = count
    
#Histograma
plt.figure(figsize=(12, 8))
plt.bar(list(amenities_dic.keys()), list(amenities_dic.values()))
plt.xlabel('Número de comodidades', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Frecuencia', fontsize=14, fontweight='bold', color="#333333")
plt.title('Distribución de las comodidades', fontsize=16, fontweight='bold', color="#222222")
plt.xticks( rotation=45,ha='right',fontsize=10) 
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
    
#-----------------------------
#    Bathrooms 
#-----------------------------

#Histograma
plt.figure(figsize=(9, 6))
bins = np.arange(1, 9.5, 0.5)
plt.hist(df['bathrooms'], bins=bins, edgecolor='black', color="#4C72B0", alpha=0.85, rwidth=1)
plt.xlabel('Número de Baños', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Frecuencia', fontsize=14, fontweight='bold', color="#333333")
plt.title('Distribución de la Cantidad de Baños', fontsize=16, fontweight='bold', color="#222222")
plt.xticks(np.arange(1, 9, 0.5), fontsize=12) 
plt.yticks(fontsize=12)
plt.show()

#BoxPlot
plt.figure(figsize=(7, 5))  
sns.set(style="white")  
sns.boxplot(y=df['bathrooms'], width=0.4, color="#4C72B0", 
            boxprops={'edgecolor': 'black', 'linewidth': 1.5},  
            medianprops={'color': 'red', 'linewidth': 2},  
            whiskerprops={'linewidth': 1.5},  
            capprops={'linewidth': 1.5},  
            flierprops={'marker': 'o', 'color': 'red', 'alpha': 0.6}) 

plt.ylabel('Número de Baños', fontsize=12, fontweight='bold', color="#333333")
plt.title('Distribución de Baños con Valores Atípicos', fontsize=14, fontweight='bold', color="#222222")
plt.legend(fontsize=12, loc="upper right")
plt.show()

#Diagrama de dispersion con metros cuadrados (buena idea para sustentar datos vacios)
plt.figure(figsize=(8, 6))
plt.scatter(df['bathrooms'], df['square_feet'], alpha=0.6, color="#4C72B0", linewidth=0.5)

plt.xlabel('Número de Baños', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Metros Cuadrados', fontsize=14, fontweight='bold', color="#333333")
plt.title('Relación entre Metros Cuadrados y Baños', fontsize=16, fontweight='bold', color="#222222")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#-----------------------------
#    Bedrooms
#-----------------------------

#Histograma
plt.figure(figsize=(9, 6))
bins = np.arange(1, 9.5, 0.5)
plt.hist(df['bedrooms'], bins=bins, edgecolor='black', color="#4C72B0", alpha=0.85, rwidth=1)
plt.xlabel('Número de Cuartos', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Frecuencia', fontsize=14, fontweight='bold', color="#333333")
plt.title('Distribución de la Cantidad de Cuartos', fontsize=16, fontweight='bold', color="#222222")
plt.xticks(np.arange(1, 9, 0.5), fontsize=12) 
plt.yticks(fontsize=12)
plt.show()

#BoxPlot
plt.figure(figsize=(7, 5))  
sns.set(style="white")  
sns.boxplot(y=df['bedrooms'], width=0.4, color="#4C72B0", 
            boxprops={'edgecolor': 'black', 'linewidth': 1.5},  
            medianprops={'color': 'red', 'linewidth': 2},  
            whiskerprops={'linewidth': 1.5},  
            capprops={'linewidth': 1.5},  
            flierprops={'marker': 'o', 'color': 'red', 'alpha': 0.6}) 

plt.ylabel('Número de Cuartos', fontsize=12, fontweight='bold', color="#333333")
plt.title('Distribución de Cuartos con Valores Atípicos', fontsize=14, fontweight='bold', color="#222222")
plt.legend(fontsize=12, loc="upper right")
plt.show()

#Diagrama de dispersion con metros cuadrados (buena idea para sustentar datos vacios)
plt.figure(figsize=(8, 6))
plt.scatter(df['bedrooms'], df['square_feet'], alpha=0.6, color="#4C72B0", linewidth=0.5)

plt.xlabel('Número de Cuartos', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Metros Cuadrados', fontsize=14, fontweight='bold', color="#333333")
plt.title('Relación entre Metros Cuadrados y Cuartos', fontsize=16, fontweight='bold', color="#222222")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#Diagrama de dispersion con cuartos
plt.figure(figsize=(8, 6))
plt.scatter(df['bathrooms'], df['bedrooms'], alpha=0.6, color="#4C72B0", linewidth=0.5)

plt.xlabel('Número de Baños', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Número de Cuartos', fontsize=14, fontweight='bold', color="#333333")
plt.title('Relación entre Cuartos y Baños', fontsize=16, fontweight='bold', color="#222222")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#-----------------------------
#   Pets
#-----------------------------
pets = df['pets_allowed']
print(pets.isnull().sum())

#-----------------------------
#    Price
#-----------------------------

#Histograma
plt.figure(figsize=(9, 6))
bins = np.arange(0, 10000, 100)
plt.hist(df['price'], bins=bins, edgecolor='black', color="#4C72B0", alpha=0.85, rwidth=1)
plt.xlabel('Precio', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Frecuencia', fontsize=14, fontweight='bold', color="#333333")
plt.title('Distribución de los precios', fontsize=16, fontweight='bold', color="#222222")
plt.xticks(np.arange(0, 10000, 1000), fontsize=12) 
plt.yticks(fontsize=12)
plt.show()

#Estadisticas descriptivas
price = [i for i in df['price'] if False == pd.isna(i)]
print("Media: ", st.mean(price))
print("Mediana: ", st.median(price))
print("Moda: ", st.mode(price))

print("Desviacion estandar: ", st.stdev(price))
print("Varianza: ", st.variance(price))
print(df['price'].isnull().sum())


#-----------------------------
#   Tamaño
#-----------------------------

#Histograma
plt.figure(figsize=(9, 6))
bins = np.arange(0, 5000, 100)
plt.hist(df['square_feet'], bins=bins, edgecolor='black', color="#4C72B0", alpha=0.85, rwidth=1)
plt.xlabel('Número de los pies cuadrados', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Frecuencia', fontsize=14, fontweight='bold', color="#333333")
plt.title('Distribución del tamaño', fontsize=16, fontweight='bold', color="#222222")
plt.xticks(np.arange(0, 5000, 1000), fontsize=12) 
plt.yticks(fontsize=12)
plt.show()

#Estadisticas descriptivas
tamaño = [i for i in df['square_feet'] if False == pd.isna(i)]
print("Media: ", st.mean(tamaño))
print("Mediana: ", st.median(tamaño))
print("Moda: ", st.mode(tamaño))

print("Desviacion estandar: ", st.stdev(tamaño))
print("Varianza: ", st.variance(tamaño))
print(df['square_feet'].isnull().sum())

#-----------------------------
#   Latitud y Longitud
#-----------------------------
print(df['cityname'].isnull().sum())
print(df['state'].isnull().sum())
print(df['latitude'].isnull().sum())
print(df['longitude'].isnull().sum())



cities_cat = df['state'].explode().str.strip().drop_duplicates().tolist()
cities_dic = {}
cities_dic["Null"] = df['state'].isnull().sum()

cityname = [i for i in df['state'] if False == pd.isna(i)]

for j in cities_cat:
    count =0
    if pd.isna(j):
      continue  
    for i in cityname:
        if j == i:
            count+=1
    cities_dic[j] = count
    

plt.figure(figsize=(12, 8))
plt.bar(list(cities_dic.keys()), list(cities_dic.values()))
plt.xlabel('Número de comodidades', fontsize=14, fontweight='bold', color="#333333")
plt.ylabel('Cantidad de apartamentos', fontsize=14, fontweight='bold', color="#333333")
plt.title("Fantidad de apartamentos por estad", fontsize=16, fontweight='bold', color="#222222")
plt.xticks( rotation=45,ha='right',fontsize=10) 
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
