# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:29:56 2025

@author: evert
"""

import os
import json
import numpy as np
import pandas as pd

from collections import Counter

#DATA processing:
    
# Define the list of desired navigation statuses
navigation_status_entry = ['under-way-using-engine', 
                           'moored', 
                           'fishing']

    
# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the directory containing JSON files
json_dir = os.path.join(script_dir, "raw_data_rotterdam", "raw_data_rotterdam_original")

# Dictionary to store JSON data
ais_data_dict = {}

# Load all JSON files into a dictionary
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):  # Ensure it's a JSON file
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                ais_data_dict[filename] = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")

# Convert dictionary keys to a list and shuffle them
file_keys = list(ais_data_dict.keys())
np.random.shuffle(file_keys)

# Compute split indices
total_files = len(file_keys)
train_split = int(0.65 * total_files)
test_split = int(0.85 * total_files)  # 60% train + 20% test = 80%

# Split the data
AIS_data_train = {key: ais_data_dict[key] for key in file_keys[:train_split]}
AIS_data_test = {key: ais_data_dict[key] for key in file_keys[train_split:test_split]}
AIS_data_val = {key: ais_data_dict[key] for key in file_keys[test_split:]}

# Print summary
print(f"Total files: {total_files}")
print(f"Training set: {len(AIS_data_train)} files")
print(f"Testing set: {len(AIS_data_test)} files")
print(f"Validation set: {len(AIS_data_val)} files")

print("")
print("Navigations status for all data:")
# Dictionary to store the counts of navigation statuses
navigation_status_counts = Counter()

# Loop through each file in ais_data_dict
for filename, file_data in ais_data_dict.items():
    # Loop through each entry in the "data" list
    if 'data' in file_data:
        for entry in file_data['data']:
            # Ensure "navigation" and "status" keys exist
            if 'navigation' in entry and 'status' in entry['navigation']:
                navigation_status = entry['navigation']['status']
                navigation_status_counts[navigation_status] += 1

# Print the counts of navigation statuses
for status, count in navigation_status_counts.items():
    print(f"{status}: {count}")
print("")
    
#Function to count navigation statuses
def count_navigation_status(ais_data_dict):
    navigation_status_counts = Counter()
    
    # Loop through each file in ais_data_dict
    for filename, file_data in ais_data_dict.items():
        # Loop through each entry in the "data" list
        if 'data' in file_data:
            for entry in file_data['data']:
                # Ensure "navigation" and "status" keys exist
                if 'navigation' in entry and 'status' in entry['navigation']:
                    navigation_status = entry['navigation']['status']
                    navigation_status_counts[navigation_status] += 1
                    
    return navigation_status_counts

# Count navigation statuses for train, test, and validation sets
train_status_counts = count_navigation_status(AIS_data_train)
test_status_counts = count_navigation_status(AIS_data_test)
val_status_counts = count_navigation_status(AIS_data_val)

# Print the counts for each set
print("Navigation status counts for training set:")
for status, count in train_status_counts.items():
    print(f"{status}: {count}")

print("\nNavigation status counts for testing set:")
for status, count in test_status_counts.items():
    print(f"{status}: {count}")

print("\nNavigation status counts for validation set:")
for status, count in val_status_counts.items():
    print(f"{status}: {count}")    
    


# Function to filter data based on navigation status
def filter_navigation_status(ais_data_dict, statuses):
    filtered_data = {}
    
    # Loop through each file in ais_data_dict
    for filename, file_data in ais_data_dict.items():
        # Create a new list to hold filtered entries for this file
        filtered_entries = []
        
        # Loop through each entry in the "data" list
        if 'data' in file_data:
            for entry in file_data['data']:
                # Ensure "navigation" and "status" keys exist
                if 'navigation' in entry and 'status' in entry['navigation']:
                    # Check if the status is in the list of desired statuses
                    if entry['navigation']['status'] in statuses:
                        filtered_entries.append(entry)
        
        # Store the filtered entries in the dictionary
        filtered_data[filename] = {
            'data': filtered_entries
        }
    
    return filtered_data

# Filter data for train, test, and validation sets
AIS_data_train_filtered = filter_navigation_status(AIS_data_train, navigation_status_entry)
AIS_data_test_filtered = filter_navigation_status(AIS_data_test, navigation_status_entry)
AIS_data_val_filtered = filter_navigation_status(AIS_data_val, navigation_status_entry)

# You can print the filtered data count for verification
print("")
print("Filtering data...")
print("")
print(f"Filtered training data count: {sum(len(file_data['data']) for file_data in AIS_data_train_filtered.values())}")
print(f"Filtered testing data count: {sum(len(file_data['data']) for file_data in AIS_data_test_filtered.values())}")
print(f"Filtered validation data count: {sum(len(file_data['data']) for file_data in AIS_data_val_filtered.values())}")
print("")
for i in range(len(navigation_status_entry)):
    print("Data filterd for:", navigation_status_entry[i])
    


    
    










