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
import seaborn as sns
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
cluster = None   # Choose cluster [0,1,2,3,4] or None for all clusters
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


navigation_status_mapping = {
    'Under-way-using-engine': 0,
    'Anchored': 1,
    'Not under command': 2,
    'Has restricted maneuverability': 3,
    'Ship draught is limiting its movement': 4,
    'Moored': 5,
    'Aground': 6,
    'Fishing': 7,
    'Under-way-sailing': 8
}

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
    
if cluster is None:
    x_train, x_test, x_val = np.delete(train_matrix, 3, axis=1), np.delete(test_matrix, 3, axis=1), np.delete(val_matrix, 3, axis=1)
    y_train, y_test, y_val = train_matrix[:, 3], test_matrix[:, 3], val_matrix[:, 3]
    
    full_matrix = np.vstack([train_matrix, test_matrix, val_matrix])
    column_names = ['Longitude', 'Latitude', 'Speed', 'Nav status', 'Cluster']
    df_AIS = pd.DataFrame(full_matrix, columns=column_names)

elif cluster in [0, 1, 2, 3, 4]:
    train_matrix_filtered = train_matrix[train_matrix[:, -1] == cluster]
    test_matrix_filtered = train_matrix[train_matrix[:, -1] == cluster]
    val_matrix_filtered = train_matrix[train_matrix[:, -1] == cluster]

    x_train = np.delete(train_matrix_filtered, 3, axis=1)
    x_test = np.delete(test_matrix_filtered, 3, axis=1)
    x_val = np.delete(val_matrix_filtered, 3, axis=1)

    y_train = train_matrix_filtered[:, 3]
    y_test = test_matrix_filtered[:, 3]
    y_val = val_matrix_filtered[:, 3]

    full_matrix = np.vstack([train_matrix_filtered, test_matrix_filtered, val_matrix_filtered])
    column_names = ['Longitude', 'Latitude', 'Speed', 'Nav status', 'Cluster']
    df_AIS = pd.DataFrame(full_matrix, columns=column_names)

else:
    print("Not a valid cluster number chosen")

#------------------- MAKE CLUSTER VISUAL --------------------------------------

# Replace Nav status integers with the corresponding labels
df_AIS['Nav status'] = df_AIS['Nav status'].map({v: k for k, v in navigation_status_mapping.items()})

# Group by 'Cluster' and 'Nav status', and count the occurrences
nav_status_counts = df_AIS.groupby(['Cluster', 'Nav status']).size().reset_index(name='Count')

# Create a figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Normal Bar Plot (First plot)
sns.barplot(x='Nav status', y='Count', hue='Cluster', data=nav_status_counts, ax=axes[0], ci=None)
axes[0].set_title('Nav Status Count per Cluster (Normal Scale)')
axes[0].set_xlabel('Nav Status')
axes[0].set_ylabel('Count')
axes[0].legend(title='Cluster', loc='upper right')
axes[0].grid(True)  # Add grid to the first plot

# Logarithmic Bar Plot (Second plot)
sns.barplot(x='Nav status', y='Count', hue='Cluster', data=nav_status_counts, ax=axes[1], ci=None)
axes[1].set_title('Nav Status Count per Cluster (Logarithmic Scale)')
axes[1].set_xlabel('Nav Status')
axes[1].set_ylabel('Count (Log Scale)')
axes[1].set_yscale('log')
axes[1].legend(title='Cluster', loc='upper right')
axes[1].grid(True)  # Add grid to the second plot

# Rotate x-axis labels for both plots to avoid overlap
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
#------------------- WORLD MAP ------------------------------------------

if cluster is not None:
    print("")
    print("Making cluster visuals for cluster:", cluster)
    
    print("Plotting Longitude and Latitude on World Map...")
    
    # Extract longitude and latitude from the dataframe
    lats = df_AIS['Latitude']
    lons = df_AIS['Longitude']
    
    # Create a GeoDataFrame from the longitude and latitude
    gdf = gpd.GeoDataFrame(df_AIS, 
                           geometry=gpd.points_from_xy(lons, lats), 
                           crs="EPSG:4326")  # EPSG:4326 is the standard for longitude/latitude

    # Define colors for each navigation status
    nav_status_colors = {
        0: 'blue',
        1: 'green',
        2: 'red',
        3: 'purple',
        4: 'orange',
        5: 'yellow',
        6: 'cyan',
        7: 'brown',
        8: 'pink'
    }
    
    # Assign colors to each point in your gdf based on its Nav status
    gdf['color'] = gdf['Nav status'].map(nav_status_colors)
    
    # Get the current directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the shapefile within the '110m_cultural' folder
    shapefile_path = os.path.join(script_dir, "110m_cultural", "ne_110m_admin_0_countries.shp")
    
    # Load the shapefile
    world = gpd.read_file(shapefile_path)
    
    # Filter for the United States (including Hawaii and Puerto Rico)
    us_and_territories = world[world['NAME'].isin(['United States', 'Puerto Rico'])]
    
    # Plot the filtered map (United States and Territories)
    ax = us_and_territories.plot(figsize=(10, 6), color='lightgray')
    
    # Plot the AIS data points, colored by the Nav status
    gdf.plot(ax=ax, color=gdf['color'], markersize=5, alpha=0.6, label="AIS Data Points")
    
    # Add basemap for better context (using Contextily)
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    
    # Create custom legend with navigation status labels
    handles = []
    for status, color in nav_status_colors.items():
        label = list(navigation_status_mapping.keys())[list(navigation_status_mapping.values()).index(status)]
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)
    
    # Set labels and title
    plt.title(f"Longitude and Latitude Plot for Cluster {cluster} (United States and Territories)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Add legend to the plot
    plt.legend(handles=handles, title="Navigation Status")
    
    # Show the plot
    plt.show()




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











