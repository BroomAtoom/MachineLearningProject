import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from preprocess_AIS import select_head
from sklearn.preprocessing import LabelEncoder
def main():
    print("Succesfully started the code")

    start_time = time.time()
    random_seed = 241
    np.random.seed(random_seed)
    
    #dataframe = select_head('NOAA_AIS/AIS_2020_01_01_fullycleaned.csv',percentage=50,randomize=True,seed=random_seed)
    dataframe = read_large_csv('NOAA_AIS/AIS_2020_01_01_fullycleaned.csv', showhead=False)
    
    # Splitting the data
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(dataframe)
    print(f"Size of values, X_train = {len(X_train)}, X_test = {len(X_test)}, y_train = {len(y_train)}, y_test = {len(y_test)}")

    # Training the model
    model = train_linear_regression(X_train, y_train)
    
    # Validating the model
    validation_accuracy = evaluate_model(model, X_val, y_val)
    print("Test Accuracy:", validation_accuracy) 
    
    # Testing the model
    test_accuracy = evaluate_model(model, X_test, y_test)
    print("Test Accuracy:", test_accuracy)
    print(f"Total time: {time.time() - start_time}")



def predict_linear_regression(model, X_test):
    """
    Make predictions using a trained linear regression model.

    Parameters:
    model (LinearRegression): Trained linear regression model.
    X_test (array-like): Test data features.

    Returns:
    numpy.ndarray: Predicted target values.
    """
    # Make predictions
    predictions = model.predict(X_test)

    return predictions

def split_data(data, test_size=0.2, val_size=0.2, random_state=241):
    """
    Split the dataset into training, testing, and validation sets.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing the dataset.
    test_size (float): Proportion of the dataset to include in the test split.
    val_size (float): Proportion of the dataset to include in the validation split.
    random_state (int): Random seed for reproducibility.

    Returns:
    Tuple: Training, testing, and validation sets.
    """
    # Select features and target
    X = data[['LAT', 'LON', 'SOG']]
    y = data['Status'].astype(int).values  # Ensure NumPy array

    print("Splitting the dataset...")
    # Split the data into training and temporary sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Split the temporary set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model using scikit-learn.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data target values.

    Returns:
    LinearRegression: Trained linear regression model.
    """
    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the accuracy of the model.

    Parameters:
    model: Trained model.
    X_test (array-like): Test data features.
    y_test (array-like): Test data target values.

    Returns:
    float: Accuracy of the model.
    """
    # Make predictions using the existing function
    print("Evaluating the model...")
    predictions = predict_linear_regression(model, X_test)

    # Round predictions to the nearest integer
    print("Rounding to the nearest integer...")
    rounded_predictions = np.round(predictions)

    # Calculate accuracy
    print("Calculating accuracy...")
    accuracy = accuracy_score(y_test, rounded_predictions)

    return accuracy

def read_large_csv(file_path,showhead=False):
    """Gives some feedback on opening and reading a large csv file"""
    print('starting to read...')
    dataframe = pd.read_csv(file_path)
    print('reading done')
    if showhead == True:
        print(dataframe.head(10))
    return dataframe


def makefolder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

if __name__ == '__main__':
    main()