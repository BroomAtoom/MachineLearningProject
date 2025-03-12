"""This file is used to combine the json files from the different datasets, and create a new json file with all the data."""
import json
import os
data_path = os.path.join('sample_raw_data_rotterdam')
output_path = os.path.join('sample_raw_data_rotterdam', 'combined.json')
combined_data = []
for file in os.listdir(data_path):
    if file.endswith('.json'):
        with open(os.path.join(data_path, file), 'r') as f:
            data = json.load(f)['data']
        combined_data.extend(data)

final_combined_json = {"data": combined_data}

with open(output_path, 'w') as outfile:
    json.dump(final_combined_json, outfile, indent=4)