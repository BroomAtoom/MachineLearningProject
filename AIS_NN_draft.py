# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:29:56 2025

@author: evert
"""

print("Importing modules...")

import os
import json
import time
import warnings
import psutil
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def memory_usage():
    """Returns memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

print("Importing finished")
print("")
print("Loading data...")
#DATA processing:
    
# Define the list of desired navigation statuses
navigation_status_entry = ['under-way-using-engine', 
                           'moored', 
                           'fishing']

    
# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the directory containing JSON files
json_dir = os.path.join(script_dir, "raw_data_rotterdam", "raw_data_rotterdam_original")

# Dictionary to store JSON data
ais_data_dict = {}

# Load all JSON files into a dictionary
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):  # Ensure it's a JSON file
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                ais_data_dict[filename] = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")

# Convert dictionary keys to a list and shuffle them
file_keys = list(ais_data_dict.keys())
np.random.shuffle(file_keys)

# Compute split indices
total_files = len(file_keys)
train_split = int(0.65 * total_files)
test_split = int(0.85 * total_files)  # 60% train + 20% test = 80%

# Split the data
AIS_data_train = {key: ais_data_dict[key] for key in file_keys[:train_split]}
AIS_data_test = {key: ais_data_dict[key] for key in file_keys[train_split:test_split]}
AIS_data_val = {key: ais_data_dict[key] for key in file_keys[test_split:]}

# Print summary
print(f"Total files: {total_files}")
print(f"Training set: {len(AIS_data_train)} files")
print(f"Testing set: {len(AIS_data_test)} files")
print(f"Validation set: {len(AIS_data_val)} files")

print("")
print("Navigations status for all data:")
# Dictionary to store the counts of navigation statuses
navigation_status_counts = Counter()

# Loop through each file in ais_data_dict
for filename, file_data in ais_data_dict.items():
    # Loop through each entry in the "data" list
    if 'data' in file_data:
        for entry in file_data['data']:
            # Ensure "navigation" and "status" keys exist
            if 'navigation' in entry and 'status' in entry['navigation']:
                navigation_status = entry['navigation']['status']
                navigation_status_counts[navigation_status] += 1

# Print the counts of navigation statuses
for status, count in navigation_status_counts.items():
    print(f"{status}: {count}")
print("")
    
#Function to count navigation statuses
def count_navigation_status(ais_data_dict):
    navigation_status_counts = Counter()
    
    # Loop through each file in ais_data_dict
    for filename, file_data in ais_data_dict.items():
        # Loop through each entry in the "data" list
        if 'data' in file_data:
            for entry in file_data['data']:
                # Ensure "navigation" and "status" keys exist
                if 'navigation' in entry and 'status' in entry['navigation']:
                    navigation_status = entry['navigation']['status']
                    navigation_status_counts[navigation_status] += 1
                    
    return navigation_status_counts

# Count navigation statuses for train, test, and validation sets
train_status_counts = count_navigation_status(AIS_data_train)
test_status_counts = count_navigation_status(AIS_data_test)
val_status_counts = count_navigation_status(AIS_data_val)

# Print the counts for each set
print("Navigation status counts for training set:")
for status, count in train_status_counts.items():
    print(f"{status}: {count}")

print("\nNavigation status counts for testing set:")
for status, count in test_status_counts.items():
    print(f"{status}: {count}")

print("\nNavigation status counts for validation set:")
for status, count in val_status_counts.items():
    print(f"{status}: {count}")    
    

# Function to filter data based on navigation status
def filter_navigation_status(ais_data_dict, statuses):
    filtered_data = {}
    
    # Loop through each file in ais_data_dict
    for filename, file_data in ais_data_dict.items():
        # Create a new list to hold filtered entries for this file
        filtered_entries = []
        
        # Loop through each entry in the "data" list
        if 'data' in file_data:
            for entry in file_data['data']:
                # Ensure "navigation" and "status" keys exist
                if 'navigation' in entry and 'status' in entry['navigation']:
                    # Check if the status is in the list of desired statuses
                    if entry['navigation']['status'] in statuses:
                        filtered_entries.append(entry)
        
        # Store the filtered entries in the dictionary
        filtered_data[filename] = {
            'data': filtered_entries
        }
    
    return filtered_data

# Filter data for train, test, and validation sets
AIS_data_train_filtered = filter_navigation_status(AIS_data_train, navigation_status_entry)
AIS_data_test_filtered = filter_navigation_status(AIS_data_test, navigation_status_entry)
AIS_data_val_filtered = filter_navigation_status(AIS_data_val, navigation_status_entry)

# You can print the filtered data count for verification
print("")
print("Filtering data...")
print("")
print(f"Filtered training data count: {sum(len(file_data['data']) for file_data in AIS_data_train_filtered.values())}")
print(f"Filtered testing data count: {sum(len(file_data['data']) for file_data in AIS_data_test_filtered.values())}")
print(f"Filtered validation data count: {sum(len(file_data['data']) for file_data in AIS_data_val_filtered.values())}")
print("")
for i in range(len(navigation_status_entry)):
    print("Data filterd for:", navigation_status_entry[i])
    
# Define a mapping for the navigation statuses
navigation_status_mapping = {
    'under-way-using-engine': 0,
    'moored': 1,
    'fishing': 2
}

# Function to extract data and create a matrix
def create_matrix(ais_data_dict):
    matrix_data = []
    
    # Loop through each file in ais_data_dict
    for filename, file_data in ais_data_dict.items():
        # Loop through each entry in the "data" list
        if 'data' in file_data:
            for entry in file_data['data']:
                # Ensure "navigation" and "location" keys exist
                if 'navigation' in entry and 'location' in entry['navigation']:
                    # Extract speed, latitude, and longitude
                    speed = entry['navigation'].get('speed', 0)  # Default to 0 if no speed
                    lat = entry['navigation']['location'].get('lat', 0)  # Default to 0 if no latitude
                    long = entry['navigation']['location'].get('long', 0)  # Default to 0 if no longitude
                    
                    # Extract navigation status and map it to the value
                    nav_status = entry['navigation'].get('status', '')
                    navigation_status = navigation_status_mapping.get(nav_status, -1)  # -1 if status is unknown
                    
                    # Append to the matrix
                    matrix_data.append([long, lat, speed, navigation_status])
    
    # Convert the matrix data to a NumPy array for easier manipulation
    return np.array(matrix_data)

# Create the matrices for train, test, and validation sets
AIS_data_train_matrix = create_matrix(AIS_data_train_filtered)
AIS_data_test_matrix = create_matrix(AIS_data_test_filtered)
AIS_data_val_matrix = create_matrix(AIS_data_val_filtered)

# Print the matrices (or their shapes)
print("Training set matrix shape:", AIS_data_train_matrix.shape)
print("Testing set matrix shape:", AIS_data_test_matrix.shape)
print("Validation set matrix shape:", AIS_data_val_matrix.shape)

# Optionally, convert to DataFrame for better readability
AIS_data_train_df = pd.DataFrame(AIS_data_train_matrix, columns=["long", "lat", "speed", "navigation_status"])
AIS_data_test_df = pd.DataFrame(AIS_data_test_matrix, columns=["long", "lat", "speed", "navigation_status"])
AIS_data_val_df = pd.DataFrame(AIS_data_val_matrix, columns=["long", "lat", "speed", "navigation_status"])

#Do matrix splitting for sklearn 
x_train = AIS_data_train_matrix[:,:3]
x_test = AIS_data_test_matrix[:,:3]
x_val = AIS_data_val_matrix[:,:3]

y_train = AIS_data_train_matrix[:, 3:]
y_test = AIS_data_test_matrix[:, 3:]
y_val = AIS_data_val_matrix[:, 3:]

print("")
print("Matrices created for Sklearn")
print("")

#MACHINE LEARNING

learning_type = 'sklearn'

match learning_type:
    case 'sklearn':
        print('Sklearn is being used...')
        print("")
        
        # Initialize the model
        train_nn = MLPClassifier(hidden_layer_sizes=(100,),  # Example with 100 neurons
                                 activation='relu',
                                 solver='adam',
                                 max_iter=5,  # We want to run one iteration at a time
                                 warm_start=True,  # Keeps the previous model state to continue from last fit
                                 random_state=0)
        # Initialize best validation accuracy and best model
        best_val_accuracy = 0.0
        best_model = None
        epoch_times = []  # To store the time taken for each epoch

        # Iterate over epochs manually
        max_epochs = 10
        for epoch in range(max_epochs):
            start_time = time.time()  # Record the start time for the epoch
            print(f"\nEpoch {epoch+1}/{max_epochs}")

            # Train the model for one epoch
            train_nn.fit(x_train, y_train)

            # Predict on the validation and test set
            pred_val = train_nn.predict(x_val)
            accuracy_val = accuracy_score(y_val, pred_val)
            pred_test = train_nn.predict(x_test)
            accuracy_test = accuracy_score(y_test, pred_test)

            print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
            print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

            # Check if the current model is better than the previous best model
            if accuracy_val > best_val_accuracy:
                best_val_accuracy = accuracy_val
                best_model = train_nn  # Save the current model as the best model
                print(f"New best model found! Validation Accuracy: {accuracy_val * 100:.2f}%")

            # Calculate memory usage
            current_memory = memory_usage()
            print(f"Memory usage after epoch {epoch+1}: {current_memory:.2f} MB")

            # Record the end time for the epoch
            end_time = time.time()
            epoch_duration = end_time - start_time  # Duration of the current epoch in seconds
            epoch_times.append(epoch_duration)

            # Calculate the average time per epoch
            avg_epoch_time = np.mean(epoch_times)

            # Estimate remaining time
            remaining_epochs = max_epochs - (epoch + 1)
            remaining_time = (remaining_epochs * avg_epoch_time) / 60

            # Print the remaining time in a readable format
            print(f"Time taken for epoch {epoch+1}: {epoch_duration:.2f} seconds")
            print(f"Estimated remaining time: {remaining_time:.2f} minutes")

        # After training, save the best model
        if best_model is not None:
            model_filename = 'AIS_first_model.joblib'
            joblib.dump(best_model, model_filename)
            print(f"Best model saved to {model_filename}")
        else:
            print("No model was saved because no improvement in validation accuracy was found.")






















    
    










