

import os
import pandas as pd



filename = "AIS_2020_01_02.csv"
output_filename = "AIS_2020_01_02_cleaned.csv"
folder_name = "NOAA_AIS"
file_path = os.path.join(folder_name,filename)
output_path = os.path.join(folder_name, output_filename)



def preprocess_AIS(file_path, output_path):
    df = pd.read_csv(file_path)
    # Convert 'Status' column to numeric, forcing errors to NaN for non-numeric values
    print("Converting status column to numeric...")
    df["Status"] = pd.to_numeric(df["Status"], errors="coerce")
    print("Converting done")

    # Drop all rows where longitude, latitude, or status are Nan.
    print("Dropping rows with NaN values in LAT, LON, or Status...")
    df = df.dropna(axis=0, subset=["LAT", "LON", "Status"])
    print("Dropping done")
    
    # Filter rows where Status is between 1 and 8
    print("Filtering rows with Status between 0 and 8...")
    df = df[df["Status"].between(0, 8, inclusive="both")]
    print("Filtering done")

    #    Save the cleaned dataframe if needed
    print("Saving CSV in progres...")
    df.to_csv(output_path, index=False)
    print(f"CSV saved successfully to {output_path}!")

preprocess_AIS(file_path, output_path)


