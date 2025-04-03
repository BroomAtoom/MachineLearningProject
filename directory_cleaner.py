# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:50:28 2025

@author: evert
"""

import os
import re
import shutil

# Define the base directory
base_dir = "new_models"

# Regular expressions to extract accuracy and cluster from filenames
txt_pattern = re.compile(r"CSV_AIS_model_accuracy_([\d.]+)_cluster_\((None|\d+)\)%\.txt")
joblib_pattern = re.compile(r"CSV_AIS_first_model_accuracy_([\d.]+)%(?:_cluster_file)?\.joblib")

# Dictionary to store accuracy-cluster mappings from .txt files
accuracy_to_cluster = {}

# First, process .txt files to determine model folders
for filename in os.listdir(base_dir):
    txt_match = txt_pattern.match(filename)
    if txt_match:
        accuracy, cluster = txt_match.groups()
        cluster_str = "None" if cluster == "None" else cluster

        # Save the mapping
        accuracy_to_cluster[accuracy] = cluster_str

        # Define the target folder
        model_folder = os.path.join(base_dir, f"model_{accuracy}_cluster_{cluster_str}")
        os.makedirs(model_folder, exist_ok=True)  # Ensure folder exists

        # Move the .txt file
        src_path = os.path.join(base_dir, filename)
        dest_path = os.path.join(model_folder, filename)
        shutil.move(src_path, dest_path)

        print(f"Moved {filename} -> {dest_path}")

# Now, process .joblib files
for filename in os.listdir(base_dir):
    joblib_match = joblib_pattern.match(filename)
    if joblib_match:
        accuracy = joblib_match.group(1)

        # Check if we have a corresponding cluster value
        cluster_str = accuracy_to_cluster.get(accuracy, "None")

        # Define the target folder
        model_folder = os.path.join(base_dir, f"model_{accuracy}_cluster_{cluster_str}")
        os.makedirs(model_folder, exist_ok=True)  # Ensure folder exists

        # Move the .joblib file
        src_path = os.path.join(base_dir, filename)
        dest_path = os.path.join(model_folder, filename)
        shutil.move(src_path, dest_path)

        print(f"Moved {filename} -> {dest_path}")

print("File organization complete!")



