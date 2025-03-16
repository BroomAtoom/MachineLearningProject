# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:29:56 2025

@author: evert
"""

#link for more AIS dat:
#https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/index.html


print("Importing modules...")

import os
import json
import time
import warnings
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

start_training_time = time.time()

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set a random seed for reproducibility
random_seed = 2
np.random.seed(random_seed)

# Create the 'models' folder if it doesn't exist
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def memory_usage():
    """Returns memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

print("Importing finished")
print("")
print("Loading data...")
print("Loading CVS files...")
print("")

#DATA processing:

# Path to the directory containing the CSV files
directory_path = './COAST_NOAA_AIS_data'  # Replace with the correct path

# Dictionary to hold DataFrames
csv_data = {}

# Loop through all files in the directory
for file_name in os.listdir(directory_path):
    # Check if the file is a CSV
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory_path, file_name)
        
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Add the DataFrame to the dictionary, using the file name (without extension) as the key
        key = os.path.splitext(file_name)[0]  # Use the file name without extension as the key
        csv_data[key] = df

print("CVS data loaded!")
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
train_split = int(0.9 * total_files)
test_split = int(0.95* total_files)  # 60% train + 20% test = 80%

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

y_train = AIS_data_train_matrix[:, 3:].ravel()
y_test = AIS_data_test_matrix[:, 3:].ravel()
y_val = AIS_data_val_matrix[:, 3:].ravel()

print("")
print("Matrices created for Sklearn")
print("")

#MACHINE LEARNING

learning_type = 'none'

match learning_type:
    case 'sklearn':
        print('Sklearn is being used...')
        print("")

        # Initialize the model
        train_nn = MLPClassifier(hidden_layer_sizes=(450,),  # Example with 600 neurons
                                 activation='relu',
                                 solver='adam',
                                 max_iter=100,  # One iteration per epoch
                                 warm_start=True,  # Keeps the previous model state to continue from last fit
                                 random_state=random_seed)

        # Initialize best validation accuracy and best model
        best_val_accuracy = 0.0
        best_model = None
        epoch_times = []  # To store the time taken for each epoch
        val_accuracies = []  # To store validation accuracies for each epoch
        test_accuracies = []  # To store test accuracies for each epoch

        # Maximum number of epochs
        max_epochs = 20
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

            # Store accuracies for plotting
            val_accuracies.append(accuracy_val)
            test_accuracies.append(accuracy_test)

            # Check if the current model is better than the previous best model
            if accuracy_val > best_val_accuracy:
                best_val_accuracy = accuracy_val
                best_model = train_nn  # Save the current model as the best model
                print(f"New best model found! Validation Accuracy: {accuracy_val * 100:.2f}%")

            # Record the end time for the epoch
            end_time = time.time()
            epoch_duration = end_time - start_time  # Duration of the current epoch in seconds
            epoch_times.append(epoch_duration)

            # Calculate the average time per epoch based on previous epochs
            avg_epoch_time = np.mean(epoch_times)

            # Estimate remaining time using the average epoch time
            remaining_epochs = max_epochs - (epoch + 1)
            remaining_time = (remaining_epochs * avg_epoch_time) / 60  # In minutes
            
            # Calculate memory usage
            current_memory = memory_usage()
            print(f"Memory usage after epoch {epoch+1}: {current_memory:.2f} MB")
            # Print the remaining time in a readable format
            print(f"Time taken for epoch {epoch+1}: {epoch_duration:.2f} seconds")
            print(f"Estimated remaining time: {remaining_time:.2f} minutes")

        # After training, save the best model with accuracy in the filename
        if best_model is not None:
            # Use test accuracy for final model filename
            accuracy_test_str = f"{accuracy_test * 100:.2f}"  # Use test accuracy instead of validation accuracy

            # Create a filename that includes the test accuracy
            model_filename = os.path.join(model_dir, f'AIS_first_model_accuracy_{accuracy_test_str}%.joblib')
            
            # Save the model
            joblib.dump(best_model, model_filename)
            print(f"Best model saved to {model_filename}")

            # Capture the end time after the model is saved
            end_training_time = time.time()
            
            # Calculate the total time taken for training and saving the model
            total_training_time = (end_training_time - start_training_time) / 60  # Convert to minutes

            # Create the text file with model details
            txt_filename = os.path.join(model_dir, f'AIS_first_model_accuracy_{accuracy_test_str}%.txt')
            with open(txt_filename, "w") as f:
                f.write(f"Random Seed: {random_seed}\n")
                f.write(f"Navigation Status Entries: {navigation_status_entry}\n")
                f.write(f"Train Percentage: {train_split / total_files * 100:.2f}%\n")
                f.write(f"Test Percentage: {(test_split - train_split) / total_files * 100:.2f}%\n")
                f.write(f"Validation Percentage: {(total_files - test_split) / total_files * 100:.2f}%\n")
                f.write(f"\nNavigation status counts for training set:\n")
                for status, count in train_status_counts.items():
                    f.write(f"{status}: {count}\n")
            
                f.write(f"\nNavigation status counts for testing set:\n")
                for status, count in test_status_counts.items():
                    f.write(f"{status}: {count}\n")
            
                f.write(f"\nNavigation status counts for validation set:\n")
                for status, count in val_status_counts.items():
                    f.write(f"{status}: {count}\n")

                f.write(f"\nMLPClassifier Parameters:\n")
                f.write(f"Hidden Layer Sizes: {train_nn.hidden_layer_sizes}\n")
                f.write(f"Activation: '{train_nn.activation}'\n")
                f.write(f"Solver: '{train_nn.solver}'\n")
                f.write(f"Max Iter: {train_nn.max_iter}\n")
                f.write(f"Warm Start: {train_nn.warm_start}\n")
                f.write(f"Random State: {train_nn.random_state}\n")
                f.write(f"\nMax Epochs: {max_epochs}\n")
                f.write(f"Model Accuracy: {accuracy_test * 100:.2f}%\n")
                f.write(f"Total Time Taken: {total_training_time:.2f} minutes\n")

            print(f"\nDetails written to {txt_filename}")
        else:
            print("No model was saved because no improvement in validation accuracy was found.")

        # Plot the validation and test accuracies over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_epochs + 1), val_accuracies, label='Validation Accuracy', color='blue')
        plt.plot(range(1, max_epochs + 1), test_accuracies, label='Test Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation and Test Accuracy per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the time taken for each epoch
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_epochs + 1), epoch_times, label='Time per Epoch', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Time Taken per Epoch')
        plt.grid(True)
        plt.show()

            



















    
    










