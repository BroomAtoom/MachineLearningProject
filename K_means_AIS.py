# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:16:39 2025

@author: evert
"""

#this is a test for k_means AIS data process

import os
import csv

# Define the folder containing the CSV files
folder_path = "/raw_data_rotterdam/CVS_files"

# Dictionary to store AIS data by ship
ais_data = {}

# Process each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Ensure it's a CSV file
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                ship_name = row["vessel.name"]
                
                # Initialize the ship's entry if not present
                if ship_name not in ais_data:
                    ais_data[ship_name] = []
                
                # Add the row data, converting appropriate fields
                ais_data[ship_name].append({
                    "draught": float(row["navigation.draught"]),
                    "time": row["navigation.time"],
                    "speed": float(row["navigation.speed"]),
                    "heading": float(row["navigation.heading"]),
                    "longitude": float(row["navigation.location.long"]),
                    "latitude": float(row["navigation.location.lat"]),
                    "course": float(row["navigation.course"]),
                    "destination": row["navigation.destination.name"],
                    "eta": row["navigation.destination.eta"],
                    "status": row["navigation.status"],
                    "mmsi": int(row["device.mmsi"]),
                    "imo": int(row["vessel.imo"]),
                    "type": row["vessel.type"]
                })

# Now, `ais_data` contains all AIS data organized by ship name
print(ais_data)
