# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:50:44 2025

@author: evert
"""

import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn import linear_model

file_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam\raw_ais_data_2021_rotterdam_1609459200.0_1609545600.0.json"

with open(file_path, "r") as file:
    data = json.load(file)

ais_records = data['data']
df = pd.json_normalize(ais_records)

# Convert time to datetime format
df["navigation.time"] = pd.to_datetime(df["navigation.time"])

# Convert numeric values
df["navigation.speed"] = pd.to_numeric(df["navigation.speed"])
df["navigation.course"] = pd.to_numeric(df["navigation.course"])
df["navigation.location.lat"] = pd.to_numeric(df["navigation.location.lat"])
df["navigation.location.long"] = pd.to_numeric(df["navigation.location.long"])

# print(df.dtypes)  # Check new data types

vessel_names = df["vessel.name"].unique()
destination = df['navigation.destination.name'].unique()
for i in range(df["vessel.name"].nunique()):
    print(f"Vessel {i} is:", vessel_names[i])

print(df["navigation.status"].value_counts())

#draft nav data for vessel ORION 

df_ORION = df[df["vessel.name"] == "ORION"]    

time_ORION = df_ORION['navigation.time']
long_ORION = df_ORION['navigation.location.long']
lat_ORION = df_ORION['navigation.location.lat']

plt.figure(1)
plt.scatter(long_ORION,lat_ORION)
plt.grid()
 
#make plot for all vessels

plt.figure(2, figsize=(10,5))
for i in range(len(vessel_names)):
    df_splitted = df[df["vessel.name"] == vessel_names[i]]  
    plt.plot(df_splitted['navigation.location.long'],df_splitted['navigation.location.lat'], label=vessel_names[i])
plt.grid()
plt.title('Ship routes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()









    