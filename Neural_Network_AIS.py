# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:27:09 2025

@author: evert
"""

print("")
print("Importing Modules...")

import os
import numpy as np
print("Import complete!")
print("")


#------------------------- LOADING DATA ---------------------------------------
print("Loading Matrices...")
# Define the parent folder and subfolder
parent_folder = "matrices"
subfolder_name = "AIS_2020_01_07_fullycleaned_top_0.5_random_seed=411"  # Change dynamically if needed
full_path = os.path.join(parent_folder, subfolder_name)

# List all .npy files in the subfolder
matrix_files = [f for f in os.listdir(full_path) if f.endswith(".npy")]

# Load each matrix and dynamically create variable names
for matrix_file in matrix_files:
    # Construct the full path to the file
    matrix_path = os.path.join(full_path, matrix_file)
    
    # Load the matrix
    matrix = np.load(matrix_path)
    
    # Create a variable name based on the file name (remove ".npy")
    variable_name = matrix_file.split(".")[0]
    
    # Dynamically assign the matrix to a variable with the same name as the file
    globals()[variable_name] = matrix

print("Matrices loaded!")
