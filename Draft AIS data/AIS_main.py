# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:14:47 2025

@author: evert
"""

import pandas as pd
import json
import os
from geopy.distance import geodesic
from JSON_compressor import merge_json_files
from JSON_reader import load_large_json_with_progress

LoadType = "PandasType"


folder_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam"
output_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam\AIS_data_compressed\AIS_compressed.json"


match LoadType:
    case 'DictType':
        # Call the function
        merge_json_files(folder_path, output_path)

        #Load JSON data and clean

        data = load_large_json_with_progress(output_path, update_interval=100)

        num_items = len(data)
        print(f"The AIS data contains {num_items} items.")


        def extract_vessel_names_and_imo_from_dict(data):
            """
            Extracts vessel names and IMO numbers from a Python dictionary representing AIS data.
            
            Parameters:
            - data (dict): A dictionary containing AIS data.
            
            Returns:
            - list: A list of tuples containing vessel name and IMO number.
            """
            vessel_info = []  # To store the (vessel_name, IMO) pairs

            # Check if the dictionary is structured as expected
            if isinstance(data, list):
                for entry in data:
                    if "data" in entry and isinstance(entry["data"], list):
                        for record in entry["data"]:
                            # Extract vessel name and IMO if present
                            vessel = record.get("vessel", {})
                            name = vessel.get("name")
                            imo = vessel.get("imo")
                            
                            if name and imo:  # Only add if both name and IMO are available
                                vessel_info.append((name, imo))
            
            return vessel_info

        print(extract_vessel_names_and_imo_from_dict(data))
    case 'PandasType':
        print("Load as PD dataframe")
        with open(output_path, "r") as file:
            json_data = json.load(file)

            # Extract vessel data (it's inside "data")
            vessel_data = json_data[0]["data"]
            
            # Normalize the nested data structure into a DataFrame
            df = pd.json_normalize(vessel_data)
            
            grouped_vessels = df.groupby("vessel.name")
            vessel_dict = {vessel: vessel_df for vessel, vessel_df in grouped_vessels}
            
            #save as CVS files
            CVS_folder = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Opdrachten_REPO\ME44312_MACHINE_LEARNING\Draft AIS data\raw_data_rotterdam\CVS_files"

            # Ensure the folder exists
            os.makedirs(CVS_folder, exist_ok=True)
            
            # Group by vessel name and save each to a separate CSV file
            vessel_dataframes = {}
            for vessel, vessel_df in df.groupby("vessel.name"):
                filename = f"{vessel.replace(' ', '_')}_AIS_data.csv"  # Replace spaces with underscores
                filepath = os.path.join(CVS_folder, filename)
                vessel_df.to_csv(filepath, index=False)
                
                if False:
                    var_name = f"vessel_{vessel.replace(' ', '_')}"
                    # Store the DataFrame in a dictionary
                    vessel_dataframes[var_name] = vessel_df
                
                    # (Optional) Store it as a variable dynamically (not recommended)
                    globals()[var_name] = vessel_df
                
            """ -----------------------------
                Use Orion as example for drift
                -----------------------------
            """
            heading_array = vessel_dict["ORION"]['navigation.heading'].to_numpy()
            course_array = vessel_dict["ORION"]['navigation.course'].to_numpy()
            drift = heading_array - course_array
            
            filtered_df = vessel_dict['ORION'].loc[vessel_dict['ORION']['navigation.speed'] < 0.2]
            
            
            
            

                

                
                
                

                
                


                
            
        
    
    














