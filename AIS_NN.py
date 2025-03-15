# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:07:37 2025

@author: evert
"""

import os
import json
import numpy as np
import pandas as pd

from collections import Counter

print("Loading JSON data...")
# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file
json_file_path = os.path.join(script_dir, "raw_data_rotterdam", "AIS_data_compressed", "AIS_compressed.json")

# Load the JSON file
with open(json_file_path, "r", encoding="utf-8") as file:
    JSON_data = json.load(file)
print("JSON data loaded")


print("Loading CVS data...")
# Define the path to the directory containing CSV files
csv_dir = os.path.join(script_dir, "raw_data_rotterdam", "CVS_files")

# Dictionary to store dataframes
AIS_data = {}

# Loop through all CSV files in the directory
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):  # Ensure it's a CSV file
        file_path = os.path.join(csv_dir, filename)
        AIS_data[filename] = pd.read_csv(file_path)  # Store DataFrame in the dictionary

print("CVS AIS data loaded in dictionary")

# Initialize a counter for navigation statuses
status_counter = Counter()

# Loop through all DataFrames in the AIS_data dictionary
for filename, df in AIS_data.items():
    # Check if 'navigation.status' exists as a column
    if "navigation.status" in df.columns:
        status_counter.update(df["navigation.status"].dropna())  # Count non-null statuses
    # If the column might be part of a multi-index dataframe
    elif ("navigation", "status") in df.columns:
        status_counter.update(df[("navigation", "status")].dropna())

# Print the counts
print("Navigation Status Counts:")
for status, count in status_counter.items():
    print(f"{status}: {count}")
    
    
    
    
    
    






