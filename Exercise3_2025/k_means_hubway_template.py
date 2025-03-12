# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from math import ceil

# Importing the dataset
dataset = pd.read_csv('hubwaytrips.csv')
# Some useful commands for you to check the dataset
print(dataset.head())  # Prints the first 5 rows of the DataFrame

dataset.info()  # DataFrame summary 

# DATA ANALYSIS
# you can analyze the data before developing the clustering method
dataset.describe()  # Statistical summary

# this step can also guide the features to be included. In this example we do not have many but in general can be useful. 
# for example what is the distribution of age for different times of the day? Is there a significant difference between them? 
NightAge = dataset[dataset["Night"] == 1]["Age"]
NightAgeAv = NightAge.mean()

EveAge = dataset[dataset["Evening"] == 1]["Age"]
EveAgeAv = EveAge.mean()

ANoonAge = dataset[dataset["Afternoon"] == 1]["Age"]
ANoonAgeAv = ANoonAge.mean()

MorAge = dataset[dataset["Morning"] == 1]["Age"]
MorAgeAv = MorAge.mean()

colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73']

plt.hist([NightAge, EveAge, ANoonAge, MorAge], density=True, color=colors, label=['Night', 'Evening', 'Afternoon', 'Morning'])
plt.title('Histogram of Age for trips across the day')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Trips')
plt.show()

# Check correlation. This gives an idea of the relation between different variables/features
fig, axes = plt.subplots(figsize=(12, 6))
correlation = dataset.corr()
corr_m = sns.heatmap(round(correlation, 2), annot=True, cmap='Blues', ax=axes, fmt='.2f')

# DATA PROCESSING
# ==== some data processing ready for you ========= #
X = dataset.iloc[:, :].values
X = pd.DataFrame(X)
# randomly select a subset of the original dataset as it is very big for the algorithm
# fix the randomness to be able to reproduce
X = X.sample(n=6000, random_state=1) 
X.columns = ['Duration', ' Morning', ' Afternoon', ' Evening', ' Night', ' Weekday', 'Weekend', 'Male', 'Age']
org_X = X  # to keep the version before feature scaling for the analysis in Part D

# Feature scaling 
X = (X - X.mean()) / X.std()
print(X.head())  # Prints the first 5 rows of the DataFrame



#  ========= PART A: Apply the K-means Algorithm step by step ====== #
print("")
print("Part A")
print("")

# Select the number of clusters #
# Here we selected three to proceed. Normally you need to decide on it through some analysis.   
k = 3

# Initialize centroids - samples k many data points randomly as the initial centroids 
centroids = X.sample(k, random_state=1)

# Function for calculating the distance
def calculate_error(a, b):
    '''
    Given two Numpy Arrays, calculates the sum of squared errors.
    '''
    error = np.sum((a - b) ** 2)
    return error 

# Function for assigning centroids 
def assign_centroids(data, centroids):
    '''
    Receives a dataframe of data and centroids and returns a list assigning each observation to a centroid.
    data: a dataframe with all data that will be used.
    centroids: a dataframe with the centroids. 
    '''
    n_observations = data.shape[0]
    centroid_assign = []
    centroid_errors = []
    k = centroids.shape[0]

    for observation in range(n_observations):
        # Calculate the error (distance) between each observation and the centroids
        errors = np.array([])
        for centroid in range(k):
            error = calculate_error(centroids.iloc[centroid, :], data.iloc[observation, :])
            errors = np.append(errors, error)

        # Assign to closest centroid
        closest_centroid = np.argmin(errors)
        centroid_error = errors[closest_centroid]

        # Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    return (centroid_assign, centroid_errors)

# Iteration loop to assign observations to centroids and update centroids
error = []
WillContinue = True
i = 0
while WillContinue:
    # Phase 1 - assign each observation to the nearest centroid
    X['centroid'], iter_error = assign_centroids(X, centroids)
    print('total error in iteration ', i, ' is: ', sum(iter_error))
    error.append(sum(iter_error))
    
    # Phase 2 - update the centroids based on the assigned observations
    centroids = X.groupby('centroid').agg('mean').reset_index(drop=True)

    # Check if the error has decreased
    if len(error) < 2:
        WillContinue = True
    else:
        if round(error[i], 3) != round(error[i - 1], 3):
            WillContinue = True
        else:
            WillContinue = False 
    i = i + 1 
print('Number of iterations:', i)

# Plot how the error evolved
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(i), error, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_title('Error vs. # Iterations')

# Final centroids and clusters
X['centroid'], iter_error = assign_centroids(X, centroids)
centroids = X.groupby('centroid').agg('mean').reset_index(drop=True)

colors = {0: 'red', 1: 'blue', 2: 'green'}

# 3D plot with final clusters and their centroids. Ploted features: Duration, Morning, and Age
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.scatter(X.iloc[:, 0], X.iloc[:, 4], X.iloc[:, 8], marker='o', c=X['centroid'].apply(lambda x: colors[x]), alpha=0.5)
ax_3d.scatter(centroids.iloc[:, 0], centroids.iloc[:, 4], centroids.iloc[:, 8], marker='o', s=300,
              c=centroids.index.map(lambda x: colors[x]))
ax_3d.set_title('Clusters of hubway trips')
ax_3d.set_xlabel('Trip duration')
ax_3d.set_ylabel('Night or not')
ax_3d.set_zlabel('Age')

plt.show()

# END OF PART A #

#  =========== PART B: Use sklearn package =========== #

print("")
print("Part B - sklearn")
print("")

# Elbow method for determining optimal number of clusters
X = X.drop('centroid', axis=1)  # remove 'centroid' column from Part A
from sklearn.cluster import KMeans
wcss = []
N_max = 30
for i in range(1, N_max):
    kmeans = KMeans(n_clusters=i, random_state=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, N_max), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('J')
plt.show()

# END OF PART B #

#  =========== PART C: Apply sklearn K-means =========== #

print("")
print("Part C - sklearn K-means")
print("")



# Apply KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=1)
X['cluster'] = kmeans.fit_predict(X)

# END OF PART C #

#  ======== PART D: RESULTS ANALYSIS ===============#

print("")
print("Part D - Results Analysis")
print("")

# Now 'cluster' is available in org_X, so we can analyze the results
org_X['cluster'] = kmeans.fit_predict(X)

# Verify if 'Night' column exists in org_X
print(org_X.columns)

# If 'Night' is missing, you can check if it exists in the original dataset
print(dataset.columns)

# Percentage of data assigned to each cluster
cluster1per = (org_X[org_X['cluster'] == 0].shape[0] / org_X.shape[0]) * 100
cluster2per = (org_X[org_X['cluster'] == 1].shape[0] / org_X.shape[0]) * 100
cluster3per = (org_X[org_X['cluster'] == 2].shape[0] / org_X.shape[0]) * 100

print(f"Cluster 1: {cluster1per:.2f}%")
print(f"Cluster 2: {cluster2per:.2f}%")
print(f"Cluster 3: {cluster3per:.2f}%")

# Average trip duration in each cluster
avgDuration1 = org_X[org_X['cluster'] == 0]['Duration'].mean()
avgDuration2 = org_X[org_X['cluster'] == 1]['Duration'].mean()
avgDuration3 = org_X[org_X['cluster'] == 2]['Duration'].mean()

print(f"Average Duration in Cluster 1: {avgDuration1:.2f}")
print(f"Average Duration in Cluster 2: {avgDuration2:.2f}")
print(f"Average Duration in Cluster 3: {avgDuration3:.2f}")

# Percentage of trips made by people above 50 years old in each cluster
perAge50_1 = (org_X[(org_X['cluster'] == 0) & (org_X['Age'] > 50)].shape[0] / org_X[org_X['cluster'] == 0].shape[0]) * 100
perAge50_2 = (org_X[(org_X['cluster'] == 1) & (org_X['Age'] > 50)].shape[0] / org_X[org_X['cluster'] == 1].shape[0]) * 100
perAge50_3 = (org_X[(org_X['cluster'] == 2) & (org_X['Age'] > 50)].shape[0] / org_X[org_X['cluster'] == 2].shape[0]) * 100

print(f"Percentage of people above 50 in Cluster 1: {perAge50_1:.2f}%")
print(f"Percentage of people above 50 in Cluster 2: {perAge50_2:.2f}%")
print(f"Percentage of people above 50 in Cluster 3: {perAge50_3:.2f}%")

# Percentage of trips made at night in each cluster
# perNI1 = (org_X[(org_X['cluster'] == 0) & (org_X['Night'] == 1)].shape[0] / org_X[org_X['cluster'] == 0].shape[0]) * 100
# perNI2 = (org_X[(org_X['cluster'] == 1) & (org_X['Night'] == 1)].shape[0] / org_X[org_X['cluster'] == 1].shape[0]) * 100
# perNI3 = (org_X[(org_X['cluster'] == 2) & (org_X['Night'] == 1)].shape[0] / org_X[org_X['cluster'] == 2].shape[0]) * 100

# print(f"Percentage of night trips in Cluster 1: {perNI1:.2f}%")
# print(f"Percentage of night trips in Cluster 2: {perNI2:.2f}%")
# print(f"Percentage of night trips in Cluster 3: {perNI3:.2f}%")
