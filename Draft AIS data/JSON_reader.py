# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:28:11 2025

@author: evert
"""
import json

def load_large_json_with_progress(file_path, update_interval):
    """
    Reads a large JSON file incrementally and prints updates every `update_interval` records.

    Parameters:
    - file_path (str): Path to the large JSON file.
    - update_interval (int): Number of records to process before printing an update.

    Returns:
    - list: List of AIS records extracted from the JSON.
    """
    records = []
    record_count = 0

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # Load the entire JSON

        # Check if the top-level structure is a list
        if isinstance(data, list):
            for entry in data:
                if "data" in entry and isinstance(entry["data"], list):
                    for record in entry["data"]:
                        records.append(record)
                        record_count += 1

                        # Print update every 'update_interval' records
                        if record_count % update_interval == 0:
                            print(f"Processed {record_count} records...")

        else:
            print("Unexpected JSON format: Expected a list at the top level.")

    print(f"Finished processing {record_count} records.")
    return records  # Returns the extracted records as a list







