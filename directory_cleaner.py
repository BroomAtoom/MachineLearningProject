# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:50:28 2025

@author: evert
"""

import os
import shutil
import re


# Path to the directory containing the models
models_dir = 'new_models'

# Function to extract accuracy and cluster from the filename
def extract_accuracy_and_cluster(filename):
    # Extract accuracy percentage
    accuracy_match = re.search(r'(\d+\.\d+)%', filename)
    accuracy = accuracy_match.group(1) if accuracy_match else None
    
    # Extract cluster value, which might be None, a number, or missing
    cluster_match = re.search(r'cluster_?\((.*?)\)', filename)
    cluster = cluster_match.group(1) if cluster_match else 'None'
    
    return accuracy, cluster

# List all files in the directory
files = os.listdir(models_dir)

# Group files by accuracy and cluster
file_groups = {}
for file in files:
    file_path = os.path.join(models_dir, file)
    
    if os.path.isfile(file_path):  # Ignore directories
        accuracy, cluster = extract_accuracy_and_cluster(file)

        if accuracy:
            accuracy = f'{float(accuracy):.2f}'  # Standardize accuracy formatting
            
            if accuracy not in file_groups:
                file_groups[accuracy] = {}

            if cluster not in file_groups[accuracy]:
                file_groups[accuracy][cluster] = []

            file_groups[accuracy][cluster].append(file)

# Process each group and move files to their respective folders
for accuracy, clusters in file_groups.items():
    for cluster, grouped_files in clusters.items():
        # Define folder name: model_{accuracy}_cluster_{cluster}
        folder_name = f'model_{accuracy}_cluster_{cluster}'
        folder_path = os.path.join(models_dir, folder_name)

        # Create the subfolder
        os.makedirs(folder_path, exist_ok=True)

        # Move the grouped files into the respective folder
        for file in grouped_files:
            source_path = os.path.join(models_dir, file)
            destination_path = os.path.join(folder_path, file)
            shutil.move(source_path, destination_path)

print("All files have been organized into their respective subfolders.")

