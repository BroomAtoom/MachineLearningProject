# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:27:09 2025

@author: evert
"""

print("")
print("Importing Modules...")

import os
import re
import time
import warnings
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

print("Import complete!")
print("")

#----------------------- INITIAL ---------------------------------------------

subfolder_name = "AIS_2020_01_09_fullycleaned_top_0.5_random_seed=7777" 
cluster = 0   # Choose cluster [0,1,2,3,4] or None for all clusters
learning_type = 'none'
data_type = 'CSV'

if cluster == None:
    print("Not filtered on cluster")
    print("All data used")
    print("")

else:
    print("Using only cluster", cluster)
    print("")


plt.rcParams['agg.path.chunksize'] = 10000

# Create the 'new_models' folder if it doesn't exist
model_dir = 'new_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

start_training_time = time.time()                               #Start time to track total run time
warnings.filterwarnings("ignore", category=ConvergenceWarning)  #Disable iter=1 warning in Sklearn

# Make function to track memory usage (used in the Machine Learning part)
def memory_usage():
    """Returns memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

train_matrix = []
test_matrix = []
val_matrix = []

x_train = []    
x_test = [] 
x_val = [] 

y_train = []
y_test = []
y_val = []

#------------------------- LOADING DATA ---------------------------------------

print("Loading Matrices...")

# Define the parent folder and subfolder
parent_folder = "matrices"
 # Change dynamically if needed
full_path = os.path.join(parent_folder, subfolder_name)

# List all .npy files in the subfolder
matrix_files = [f for f in os.listdir(full_path) if f.endswith(".npy")]

# Load each matrix and dynamically create variable names
for matrix_file in matrix_files:
    # Construct the full path to the file
    matrix_path = os.path.join(full_path, matrix_file)
    
    # Load the matrix
    matrix = np.load(matrix_path, allow_pickle=True)
    
    # Create a variable name based on the file name (remove ".npy")
    variable_name = matrix_file.split(".")[0]
    
    # Dynamically assign the matrix to a variable with the same name as the file
    globals()[variable_name] = matrix

print("Matrices loaded!")

# Load the models
print("Loading models...")
for root, dirs, files in os.walk(full_path):  # Walk only through the current subfolder
    for file in files:
        if file.endswith('.joblib'):  # Only process .joblib files
            model_path = os.path.join(root, file)
            print(f"Loading model from {model_path}")
            K_model = joblib.load(model_path)

print("K_model loaded!")

#--------------------- FINDING RANDOM SEED WITH REGEX -------------------------

# Define the regex pattern to extract the value of random_seed
pattern = r"random_seed=(\d+)"

# Search for the pattern
match = re.search(pattern, subfolder_name)

# Check if the pattern is found and extract the value
if match:
    random_seed = int(match.group(1))
    print(f"Random Seed Value: {random_seed}")
else:
    print("Random seed not found")


    
#--------------------- EXTRACT DATA FOR ONE CLUSTER ---------------------------    
    
if cluster == None:

    x_train = np.delete(train_matrix, 3, axis=1)
    x_test = np.delete(test_matrix, 3, axis=1)
    x_val = np.delete(val_matrix, 3, axis=1)
    
    y_train = train_matrix[:, 3]
    y_test = test_matrix[:, 3]
    y_val = val_matrix[:, 3]   
    
elif cluster == 0:
    
    train_matrix_filtered = train_matrix[train_matrix[:, -1] == 0]
    test_matrix_filtered = train_matrix[train_matrix[:, -1] == 0]
    val_matrix_filtered = train_matrix[train_matrix[:, -1] == 0]
    
    x_train = np.delete(train_matrix_filtered, 3, axis=1)
    x_test = np.delete(test_matrix_filtered, 3, axis=1)
    x_val = np.delete(val_matrix_filtered, 3, axis=1)
    
    y_train = train_matrix_filtered[:, 3]
    y_test = test_matrix_filtered[:, 3]
    y_val = val_matrix_filtered[:, 3] 
    
elif cluster == 1:
    
    train_matrix_filtered = train_matrix[train_matrix[:, -1] == 1]
    test_matrix_filtered = train_matrix[train_matrix[:, -1] == 1]
    val_matrix_filtered = train_matrix[train_matrix[:, -1] == 1]
    
    x_train = np.delete(train_matrix_filtered, 3, axis=1)
    x_test = np.delete(test_matrix_filtered, 3, axis=1)
    x_val = np.delete(val_matrix_filtered, 3, axis=1)
    
    y_train = train_matrix_filtered[:, 3]
    y_test = test_matrix_filtered[:, 3]
    y_val = val_matrix_filtered[:, 3]   
    
elif cluster == 2:
    
    train_matrix_filtered = train_matrix[train_matrix[:, -1] == 2]
    test_matrix_filtered = train_matrix[train_matrix[:, -1] == 2]
    val_matrix_filtered = train_matrix[train_matrix[:, -1] == 2]
    
    x_train = np.delete(train_matrix_filtered, 3, axis=1)
    x_test = np.delete(test_matrix_filtered, 3, axis=1)
    x_val = np.delete(val_matrix_filtered, 3, axis=1)
    
    y_train = train_matrix_filtered[:, 3]
    y_test = test_matrix_filtered[:, 3]
    y_val = val_matrix_filtered[:, 3]  
    
elif cluster == 3:
    
    train_matrix_filtered = train_matrix[train_matrix[:, -1] == 3]
    test_matrix_filtered = train_matrix[train_matrix[:, -1] == 3]
    val_matrix_filtered = train_matrix[train_matrix[:, -1] == 3]
    
    x_train = np.delete(train_matrix_filtered, 3, axis=1)
    x_test = np.delete(test_matrix_filtered, 3, axis=1)
    x_val = np.delete(val_matrix_filtered, 3, axis=1)
    
    y_train = train_matrix_filtered[:, 3]
    y_test = test_matrix_filtered[:, 3]
    y_val = val_matrix_filtered[:, 3] 

elif cluster == 4:
    
    train_matrix_filtered = train_matrix[train_matrix[:, -1] == 4]
    test_matrix_filtered = train_matrix[train_matrix[:, -1] == 4]
    val_matrix_filtered = train_matrix[train_matrix[:, -1] == 4]
    
    x_train = np.delete(train_matrix_filtered, 3, axis=1)
    x_test = np.delete(test_matrix_filtered, 3, axis=1)
    x_val = np.delete(val_matrix_filtered, 3, axis=1)
    
    y_train = train_matrix_filtered[:, 3]
    y_test = test_matrix_filtered[:, 3]
    y_val = val_matrix_filtered[:, 3] 
    
else:
    print("Not a valid cluster number chosen")
    
    
#------------------- CLUSTER VISUALS ------------------------------------------

if not cluster == None:
    print("")
    print("Making cluster visuals for cluster:", cluster)
    full_matrix = np.vstack([train_matrix_filtered, 
                             test_matrix_filtered, 
                             val_matrix_filtered])
    column_names = ['Longitude', 'Latitude', 'Speed', 'Nav status', 'Cluster']
    df_AIS = pd.DataFrame(full_matrix, columns=column_names)
    


#------------------ MACHINE LEARNING ------------------------------------------

match learning_type:
    case 'sklearn':
        print('Sklearn is being used...')
        print("")

        # Initialize the model
        train_nn = MLPClassifier(hidden_layer_sizes=(4,8,8,4),
                                 activation = "relu",
                                 solver='adam',
                                 max_iter=60,  
                                 warm_start= True,  # Keeps the previous model state to continue from last fit
                                 random_state=random_seed)
        print(train_nn.get_params())
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
            model_filename = os.path.join(model_dir, f'{data_type}_AIS_first_model_accuracy_{accuracy_test_str}%.joblib')
            model_filename_K_means = os.path.join(model_dir, f'{data_type}_AIS_first_model_accuracy_{accuracy_test_str}%_cluster_file.joblib')
            # Save the model
            joblib.dump(best_model, model_filename)
            joblib.dump(K_model, model_filename_K_means)
            print(f"Best model saved to {model_filename}")

            # Capture the end time after the model is saved
            end_training_time = time.time()
            
            # Calculate the total time taken for training and saving the model
            total_training_time = (end_training_time - start_training_time) / 60  # Convert to minutes
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
            # Create the text file with model details
            txt_filename = os.path.join(model_dir, f'{data_type}_AIS_model_accuracy_{accuracy_test_str}_cluster_({cluster})%.txt')
            with open(txt_filename, "w") as f:
                f.write(f"Date and Time: {current_time}\n")
                f.write(f"Data used: {subfolder_name}\n")
                f.write(f"Data used: {data_type}\n")
                f.write(f"Cluster used: {cluster}\n")
                # if clustering == 'labels':
                #     f.write(f"Data clustering used: {clustering}\n")
                #     f.write(f"Number of clusters: {n_clusters}\n")
                # elif clustering == 'distance':
                #     f.write(f"Data clustering used: {clustering}\n")
                #     f.write(f"Number of clusters: {n_clusters}\n")
                # else:
                #     f.write(f"Data clustering used: None \n")
                f.write(f"Random Seed: {random_seed}\n")
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











