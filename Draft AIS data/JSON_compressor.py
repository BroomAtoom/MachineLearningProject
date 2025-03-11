# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:14:47 2025

@author: evert
"""

import pandas as pd
import json
import os

def merge_json_files(folder_path, output_path):
    """
    Reads all JSON files in a folder, merges them into a single Pandas DataFrame, 
    and saves the result as a JSON file.

    Parameters:
    - folder_path (str): Path to the folder containing JSON files.
    - output_path (str): Path to save the final merged JSON file.
    """
    # Get list of all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    df_list = []
    for file in json_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # Load JSON data
            df = pd.json_normalize(data)  # Flatten nested JSON if needed
            df_list.append(df)

    # Concatenate all DataFrames
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_json(output_path, orient="records", indent=4)
        print(f"JSON files merged successfully and saved to: {output_path}")
    else:
        print("No JSON files found in the folder.")

# Example usage
folder_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam"
output_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam\AIS_data_compressed\AIS_compressed.json"

merge_json_files(folder_path, output_path)
