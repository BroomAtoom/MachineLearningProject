import os
import json
import datetime

# Define the source directory and the new folder name
source_directory = r'C:\Users\jarri\Desktop\Spyder_Conda\bigproject\sample_raw_data_rotterdam'  # Change this to your directory
new_folder = 'new_pretty_printed_json'

# Create a new folder path
new_folder_path = os.path.join(source_directory, new_folder)

# Create the new directory if it does not exist
os.makedirs(new_folder_path, exist_ok=True)

# Iterate through all files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.json'):
        file_path = os.path.join(source_directory, filename)
        
        # Read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Extract the Unix timestamps from the filename
        parts = filename.split('_')
        start_timestamp = float(parts[-2])  # Second to last part
        end_timestamp = float(parts[-1].replace('.json', ''))  # Last part, remove .json
        
        # Convert timestamps to a readable format
        start_date = datetime.datetime.fromtimestamp(start_timestamp).strftime("%Y%m%d_%H%M%S")
        end_date = datetime.datetime.fromtimestamp(end_timestamp).strftime("%Y%m%d_%H%M%S")

        # Create a new filename using the formatted timestamps
        new_filename = f"{'_'.join(parts[:-2])}_{start_date}_{end_date}.json"  # Replace the last two parts with the new dates
        pretty_file_path = os.path.join(new_folder_path, new_filename)
        
        # Pretty print the JSON data and save it to the new folder
        with open(pretty_file_path, 'w') as pretty_json_file:
            json.dump(data, pretty_json_file, indent=4)  # Use indent for pretty printing

print(f"Pretty printed JSON files have been saved to: {new_folder_path}")