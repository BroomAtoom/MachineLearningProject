# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:50:57 2025

@author: evert
"""

import pandas as pd
import json 
import os

folder_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam"
output_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam\AIS_data_compressed\AIS_compressed.json"


AIS_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

df_list = []
for file in AIS_files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Load JSON data
        df = pd.json_normalize(data)  # Flatten nested JSON if needed
        df_list.append(df)

final_df = pd.concat(df_list, ignore_index=True)
final_df.to_json(output_path, orient="records", indent=4)



