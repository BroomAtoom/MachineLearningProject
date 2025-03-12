# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:16:39 2025

@author: evert
"""

# This is a test for k_means AIS data process

import os
import csv

# Define the folder path relative to the script's working directory
folder_path = os.path.join(os.getcwd(), "raw_data_rotterdam", "CVS_files")

# Dictionary to store AIS data by ship
ais_data = {}

# Process each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Ensure it's a CSV file
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                ship_name = row["vessel.name"].strip()
                
                # Initialize the ship's entry if not present
                if ship_name not in ais_data:
                    ais_data[ship_name] = []
                
                # Helper function to convert empty or invalid fields to None or default value
                def safe_float(value, default=None):
                    try:
                        return float(value) if value.strip() else default
                    except ValueError:
                        return default

                # Store AIS data for this ship with checks for empty or invalid fields
                ais_data[ship_name].append({
                    "draught": safe_float(row["navigation.draught"]),
                    "time": row["navigation.time"],
                    "speed": safe_float(row["navigation.speed"]),
                    "heading": safe_float(row["navigation.heading"], default=0.0),
                    "longitude": safe_float(row["navigation.location.long"]),
                    "latitude": safe_float(row["navigation.location.lat"]),
                    "course": safe_float(row["navigation.course"]),
                    "destination": row["navigation.destination.name"],
                    "eta": row["navigation.destination.eta"],
                    "status": row["navigation.status"],
                    "mmsi": int(row["device.mmsi"]),
                    "imo": int(row["vessel.imo"]),
                    "type": row["vessel.type"]
                })

# Print an example of the structured data
for ship, records in ais_data.items():
    print(f"Ship: {ship}, Number of records: {len(records)}")
    print(records[:2])  # Print first two records for preview
    break  # Just to limit output

# Now, `ais_data` contains all AIS data organized by ship name


#data finished





