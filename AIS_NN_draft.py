# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:29:56 2025

@author: evert
"""

#link for more AIS dat:
#https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/index.html


#TODO make new clustering
#TODO determine number of clusters
#TODO make validations
#TODO make linear regression

print("Importing modules...")

import os
import json
import time
import warnings
import psutil
import joblib
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans

print("Modules imported!")

#------------------ INPUT PARAMETERS ------------------------------------------

learning_type = 'none'
random_seed = 341

# Wich data to train? 'JSON', 'CSV', or 'Both'
data_type = 'CSV'

clustering = 'distance'

#------------------ INITIAL CODE-----------------------------------------------


start_training_time = time.time()                               #Start time to track total run time

warnings.filterwarnings("ignore", category=ConvergenceWarning)  #Disable iter=1 warning in Sklearn

np.random.seed(random_seed)

match data_type:
    case 'JSON':
        print("")
        print("Using: JSON data")
        print("")
    case 'CSV':
        print("")
        print("Using: CSV data")
        print("")
    case 'Both':
        print("")
        print("Using: JSON and CSV data")
        print("")
        
if data_type not in ['JSON', 'CSV', 'Both']:
    print("Select: JSON, CSV or Both:")
    data_type = input()
    print("Selected:", data_type)
    if data_type not in ['JSON', 'CSV', 'Both']:
        print("Invalid data type! Exiting the program.")
        sys.exit()


# Create the 'models' folder if it doesn't exist
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Make function to track memory usage (used in the Machine Learning part)
def memory_usage():
    """Returns memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

# ------------------ NAV MAPPING ----------------------------------

# Define the list of desired navigation statuses
navigation_status_entry = ['under-way-using-engine', 
                           'moored', 
                           'fishing']

# Define a mapping for the navigation statuses (the one that actually will be used)
navigation_status_mapping = {'under-way-using-engine': 0,
    'anchored': 1,
    'Not under command': 2,
    'Has restricted maneuverability': 3,
    'Ship draught is limiting its movement': 4,
    'moored': 5,
    'aground': 6,
    'fishing': 7,
    'under-way-sailing': 8
    }

# Full nav encoding (https://coast.noaa.gov/data/marinecadastre/ais/data-dictionary.pdf)
navigation_status_mapping_full = {
    'under-way-using-engine': 0,
    'anchored': 1,
    'Not under command': 2,
    'Has restricted maneuverability': 3,
    'Ship draught is limiting its movement': 4,
    'moored': 5,
    'aground': 6,
    'fishing': 7,
    'under-way-sailing': 8,
    'reserved code': 9,
    'towing': 10,
    'search and rescue': 14,
    'undefined': 15
}    

# ------------------ DATA PROCESSING ----------------------------------

print("Importing finished\n")
print("Loading data...")
print("Loading CSV files...\n")

# CSV AIS DATA:

# Path to the directory containing the CSV files
directory_path = './AIS_DATA_CLEANED'  # Replace with the correct path

# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get list of CSV files
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
total_files = len(csv_files)  # Count total CSV files

# Dictionary to hold DataFrames
csv_data = {}

print(f"Found {total_files} CSV files. Starting to load...\n")

# Loop through all files and track progress
for i, file_name in enumerate(csv_files):
    file_path = os.path.join(directory_path, file_name)
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Add the DataFrame to the dictionary
    key = os.path.splitext(file_name)[0]  # Use filename without extension as the key
    csv_data[key] = df

    # Print progress at every 10% milestone
    progress = (i + 1) / total_files * 100
    if progress % 10 < (100 / total_files) or i == total_files - 1:  # Print at ~10%, 20%, ..., 100%
        print(f"Loading progress: {int(progress)}% complete")

print("\nCSV data loaded!")
print("")
print("Finding NAV status values...")

# Reverse the mapping for easy lookup
status_description_mapping = {v: k for k, v in navigation_status_mapping_full.items()}

# Initialize counter
status_counter_CSV = Counter()

# Iterate through all filenames in csv_data
for filename, data in csv_data.items():
    if "Status" in data:
        for status in data["Status"]:
            # Convert to integer if possible, handling NaN values
            if isinstance(status, (int, float)) and not np.isnan(status):
                status_counter_CSV[int(status)] += 1  # Ensure integer keys

# Print the counts with descriptions
for status in range(16):  # Ensure all statuses are checked
    count = status_counter_CSV.get(status, 0)
    description = status_description_mapping.get(status, f"Unknown status {status}")
    print(f"There are {count} cases of status {status} ({description})")

print("\nFiltering CSV data on nav status...")

# ------------------ FILTERING CSV DATA ----------------------------------

# Define a set of allowed navigation status values
allowed_statuses = set(navigation_status_mapping.values())

# Initialize the filtered dictionary
CSV_data_filtered = {}

# Iterate through all filenames in csv_data
for filename, data in csv_data.items():
    if "Status" in data:
        # Convert to NumPy array for efficient filtering
        status_array = np.array(data["Status"])

        # Create a Boolean mask where status is in allowed_statuses
        mask = np.isin(status_array, list(allowed_statuses))

        # If there are matching entries, filter all columns efficiently
        if np.any(mask):  # Only process if at least one match exists
            CSV_data_filtered[filename] = {
                key: np.array(values)[mask].tolist() for key, values in data.items()
            }
        print("Filtering iteration done!")

print("Filtering complete!")
print("Splitting CSV data...")
# Initialize dictionaries for training, testing, and validation data
CSV_data_train = {}
CSV_data_test = {}
CSV_data_val = {}

# Set the random seed for reproducibility (optional)
random.seed(random_seed)

# Loop through each filename in the filtered data
for filename, data in CSV_data_filtered.items():
    # Get the number of rows in the current dataset
    num_rows = len(data['Status'])  # Assuming 'Status' is present in all rows
    
    # Generate a list of indices to shuffle
    indices = list(range(num_rows))
    
    # Shuffle the indices randomly
    random.shuffle(indices)
    
    # Split indices into 80%, 10%, 10%
    split_80 = int(0.8 * num_rows)
    split_90 = int(0.9 * num_rows)
    
    train_indices = indices[:split_80]
    test_indices = indices[split_80:split_90]
    val_indices = indices[split_90:]
    
    # Create subsets for training, testing, and validation
    CSV_data_train[filename] = {key: [values[i] for i in train_indices] for key, values in data.items()}
    CSV_data_test[filename] = {key: [values[i] for i in test_indices] for key, values in data.items()}
    CSV_data_val[filename] = {key: [values[i] for i in val_indices] for key, values in data.items()}

# Print the size of each set to verify
print("")
print(f"Training set size: {sum(len(data['Status']) for data in CSV_data_train.values())} rows")
print(f"Testing set size: {sum(len(data['Status']) for data in CSV_data_test.values())} rows")
print(f"Validation set size: {sum(len(data['Status']) for data in CSV_data_val.values())} rows")


# -------------------------------- JSON AIS DATA ---------------------------------------
match data_type:
    case 'JSON':
        print("")
        print("Loading JSON data...")   
        print("") 
            
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
        train_split = int(0.8 * total_files)
        test_split = int(0.9* total_files)  # 60% train + 20% test = 80%
        
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


#-------------------- CREATING TRAIN, TEST AND VALIDATION MATRICES ----------------
print("")
print("Creating matrices from", data_type, "for Sklearn...")

# Function to extract data and create a matrix for JSON files
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


def create_matrix_from_csv(CSV_data_dict):
    matrix_data = []
    total_entries = sum(len(file_data['Status']) for file_data in CSV_data_dict.values() if 'Status' in file_data)
    processed_entries = 0
    last_printed_percentage = -5  # Ensures the first update starts at 0%

    # Iterate through all files in CSV_data_dict
    for filename, file_data in CSV_data_dict.items():
        if 'Status' in file_data and 'LAT' in file_data and 'LON' in file_data and 'SOG' in file_data:
            num_entries = len(file_data['Status'])

            for i in range(num_entries):
                speed = file_data['SOG'][i] if not np.isnan(file_data['SOG'][i]) else 0
                lat = file_data['LAT'][i] if not np.isnan(file_data['LAT'][i]) else 0
                long = file_data['LON'][i] if not np.isnan(file_data['LON'][i]) else 0
                nav_status = file_data['Status'][i]

                matrix_data.append([long, lat, speed, nav_status])
                
                # Update progress tracking
                processed_entries += 1
                percentage_done = (processed_entries / total_entries) * 100
                
                if percentage_done >= last_printed_percentage + 5:
                    print(f"Progress: {percentage_done:.1f}%")
                    last_printed_percentage += 5

    return np.array(matrix_data)

def count_last_column_values(matrix):
    """
    Counts occurrences of values in the last column of a 2D matrix 
    or directly in a 1D list, and prints the results.
    
    Args:
        matrix (list or np.ndarray): A list of lists (2D matrix) or a single list (1D array).
    """
    # Ensure the input is a Python list
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()

    # If it's a 2D matrix, extract the last column
    if isinstance(matrix[0], list):  
        values = [row[-1] for row in matrix]
    else:
        values = matrix  # It's a 1D list

    # Ensure values is a list of hashable elements
    values = list(map(int, values))  # Convert to int to avoid unhashable errors

    # Count occurrences
    count = Counter(values)

    # Print the result
    for value, freq in count.items():
        print(f"{value} = {freq} times")

        

match data_type:
    case 'JSON':
        # Create the matrices for train, test, and validation sets
        AIS_data_train_matrix = create_matrix(AIS_data_train_filtered)
        AIS_data_test_matrix = create_matrix(AIS_data_test_filtered)
        AIS_data_val_matrix = create_matrix(AIS_data_val_filtered)
        
        print("Training set matrix shape JSON data:", AIS_data_train_matrix.shape)
        print("Testing set matrix shape JSON data:", AIS_data_test_matrix.shape)
        print("Validation set matrix shape JSON data:", AIS_data_val_matrix.shape)
        
        #Do matrix splitting for sklearn 
        x_train = AIS_data_train_matrix[:,:3]
        x_test = AIS_data_test_matrix[:,:3]
        x_val = AIS_data_val_matrix[:,:3]
        
        y_train = AIS_data_train_matrix[:, 3:].ravel()
        y_test = AIS_data_test_matrix[:, 3:].ravel()
        y_val = AIS_data_val_matrix[:, 3:].ravel()
        
        print("")
        print("Matrices created for Sklearn with JSON data")
        print("")
        
    case 'CSV':

        #Do matrix splitting for sklearn
        print("Creating X_train...")
        CSV_AIS_data_train_matrix = create_matrix_from_csv(CSV_data_train)
        count_last_column_values(CSV_AIS_data_train_matrix)
        x_train = CSV_AIS_data_train_matrix[:,:3]
        print("     X_train created!")
        print("Creating X_test...")
        
        CSV_AIS_data_test_matrix = create_matrix_from_csv(CSV_data_test)
        count_last_column_values(CSV_AIS_data_test_matrix)
        x_test = CSV_AIS_data_test_matrix[:,:3]
        print("     X_test created!")
        print("Creating X_val...")
        CSV_AIS_data_val_matrix = create_matrix_from_csv(CSV_data_val)
        count_last_column_values(CSV_AIS_data_val_matrix)
        x_val = CSV_AIS_data_val_matrix[:,:3]
        print("     X_val created!")
        
        print("Training set matrix shape CSV data:", CSV_AIS_data_train_matrix.shape)
        print("Testing set matrix shape CSV data:", CSV_AIS_data_test_matrix.shape)
        print("Validation set matrix shape CSV data:", CSV_AIS_data_val_matrix.shape)

        y_train = CSV_AIS_data_train_matrix[:, 3:].ravel()
        y_test = CSV_AIS_data_test_matrix[:, 3:].ravel()
        y_val = CSV_AIS_data_val_matrix[:, 3:].ravel()
        
        print("")
        print("Checking for amount of nav statuses in y_train:")
        count_last_column_values(y_train)
        print("")
        print("Checking for amount of nav statuses in y_test:")
        count_last_column_values(y_test)
        print("")
        print("Checking for amount of nav statuses in y_val:")
        count_last_column_values(y_val)
        print("")
        print("Matrices created for Sklearn with CSV data")
        print("")
    # case 'Both':
    #     print("Training set matrix shape JSON data:", AIS_data_train_matrix.shape)
    #     print("Testing set matrix shape JSON data:", AIS_data_test_matrix.shape)
    #     print("Validation set matrix shape JSON data:", AIS_data_val_matrix.shape)
    #     print("")
    #     print("Training set matrix shape CSV data:", CSV_AIS_data_train_matrix.shape)
    #     print("Testing set matrix shape CSV data:", CSV_AIS_data_test_matrix.shape)
    #     print("Validation set matrix shape CSV data:", CSV_AIS_data_val_matrix.shape)
        
    #     #Do matrix splitting for sklearn 
    #     x_train = np.vstack((CSV_AIS_data_train_matrix[:,:3],AIS_data_train_matrix[:,:3]))
    #     x_test = np.vstack((CSV_AIS_data_test_matrix[:,:3],AIS_data_test_matrix[:,:3]))
    #     x_val = np.vstack((CSV_AIS_data_val_matrix[:,:3],AIS_data_val_matrix[:,:3]))
        
    #     # y_train = np.vstack((CSV_AIS_data_train_matrix[:, 3:].ravel(),AIS_data_train_matrix[:, 3:].ravel())) 
    #     # y_test = np.vstack((CSV_AIS_data_test_matrix[:, 3:].ravel(),AIS_data_test_matrix[:, 3:].ravel())) 
    #     # y_val = np.vstack((CSV_AIS_data_val_matrix[:, 3:].ravel(),AIS_data_val_matrix[:, 3:].ravel())) 
    #     print("")
    #     print("Matrices created for Sklearn with JSON and CVS data")
    #     print("")


#------------------ K-MEANS DATA CLUSTERING ----------------------------------

    
print("")
print("Starting on K-means:")
print("Starting Elbow-method to determine K...")

 # 'labels' or 'distance'
n_clusters = 5      # Number of clusters (can be changed)

# Range of cluster sizes to test
k_range = range(1, 20)  # Test for 1 to 10 clusters
inertia_values = []      # List to store inertia values

# Loop through the range of cluster numbers
for k in k_range:
    # Fit K-Means on the training data for each k
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init='auto')
    kmeans.fit(x_train)  # Fit the model
    inertia_values.append(kmeans.inertia_)  # Store the inertia value

# Plot Inertia vs Number of Clusters
plt.plot(k_range, inertia_values, marker='o')
plt.grid()
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

match data_type:
    case 'CSV':
        n_clusters = 5
        print("Using K = 5 for elbow point")
    case 'JSON':
        print("Using K = 8 ")
        n_clusters = 8
        
if clustering == 'distance':
    
    print("K-means to cluster data")
    print("")
    print("K-clustering using distances...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init='auto')
    kmeans.fit(x_train)
    
    # Get distances to each cluster center
    train_distances = kmeans.transform(x_train)  
    test_distances = kmeans.transform(x_test)    
    val_distances = kmeans.transform(x_val)      
    
    # Append distances as new features
    x_train_augmented = np.hstack((x_train, train_distances))
    x_test_augmented = np.hstack((x_test, test_distances))
    x_val_augmented = np.hstack((x_val, val_distances))
    
    x_train = x_train_augmented
    x_test = x_test_augmented
    x_val = x_val_augmented
    
    print("Clustering complete!")
    print("")

else:
    print("No clustering")
    print("Using normal data")
        
print("")        
#------------------ MACHINE LEARNING -----------------------------------------


match learning_type:
    case 'sklearn':
        print('Sklearn is being used...')
        print("")
        print("Apply mapping...")
        # label_map = {0: 0, 1: 1, 5: 2, 7: 3}                          # Map original labels to 0,1,2,3
        # y_train_mapped = np.array([label_map[y] for y in y_train])  # Apply mapping to training labels
        # y_val = np.array([label_map[y] for y in y_val])             # Apply mapping to validation labels
        # y_test = np.array([label_map[y] for y in y_test])           # Apply mapping to validation labels
        print("Mapping finished!")
        print("")

        # Initialize the model
        train_nn = MLPClassifier(hidden_layer_sizes=(6,12,24,50,50,24,12,6),
                                 solver='adam',
                                 max_iter=100,  
                                 warm_start=True,  # Keeps the previous model state to continue from last fit
                                 random_state=random_seed)

        # Initialize best validation accuracy and best model
        best_val_accuracy = 0.0
        best_model = None
        epoch_times = []  # To store the time taken for each epoch
        val_accuracies = []  # To store validation accuracies for each epoch
        test_accuracies = []  # To store test accuracies for each epoch

        # Maximum number of epochs
        max_epochs = 100
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
            model_filename = os.path.join(model_dir, f'{data_type}_AIS_first_model_accuracy_{accuracy_test_str}%.joblib')
            
            # Save the model
            joblib.dump(best_model, model_filename)
            print(f"Best model saved to {model_filename}")

            # Capture the end time after the model is saved
            end_training_time = time.time()
            
            # Calculate the total time taken for training and saving the model
            total_training_time = (end_training_time - start_training_time) / 60  # Convert to minutes
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
            # Create the text file with model details
            txt_filename = os.path.join(model_dir, f'{data_type}_AIS_first_model_accuracy_{accuracy_test_str}%.txt')
            with open(txt_filename, "w") as f:
                f.write(f"Date and Time: {current_time}\n")
                f.write(f"Data used: {filename}\n")
                f.write(f"Data used: {data_type}\n")
                if clustering == 'labels':
                    f.write(f"Data clustering used: {clustering}\n")
                    f.write(f"Number of clusters: {n_clusters}\n")
                elif clustering == 'distance':
                    f.write(f"Data clustering used: {clustering}\n")
                    f.write(f"Number of clusters: {n_clusters}\n")
                else:
                    f.write(f"Data clustering used: None \n")
                f.write(f"Random Seed: {random_seed}\n")
                f.write(f"Navigation Status Entries: {navigation_status_entry}\n")
                # f.write(f"Train Percentage: {train_split / total_files * 100:.2f}%\n")
                # f.write(f"Test Percentage: {(test_split - train_split) / total_files * 100:.2f}%\n")
                # f.write(f"Validation Percentage: {(total_files - test_split) / total_files * 100:.2f}%\n")
                # f.write(f"\nNavigation status counts for training set:\n")
            
                # for status, count in train_status_counts.items():
                #     f.write(f"{status}: {count}\n")
            
                # f.write(f"\nNavigation status counts for testing set:\n")
                # for status, count in test_status_counts.items():
                #     f.write(f"{status}: {count}\n")
            
                # f.write(f"\nNavigation status counts for validation set:\n")
                # for status, count in val_status_counts.items():
                #     f.write(f"{status}: {count}\n")

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

            



















    
    










