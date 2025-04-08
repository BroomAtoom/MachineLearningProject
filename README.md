# MachineLearningProject
ME44312

Branch EvertIngwersen code uitleg

De belangrijkste scripts en folders zijn:
    - AIS_DATA_CLEANED'
    - COAST_NOAA_AIS_data
    - matrices
    - new_models
    - directory_cleaner.py
    - matrix_creator.py
    - preprocess_AIS_branchEvert.py
    - Neural_Network_AIS.py

De rest van de bestanden en mappen zijn onzin (maar verwijder maar niet voor de zekerheid)

Step-by-step:

1) AIS-CSV files are downloaded from #https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/index.html and put in the COAST_NOAA_AIS_DATA folder
2) The AIS-CSV files are loaded into the preprocess_AIS_branchEvert.py script. This script is a copy of Jarijks script (changed some stuff to make work on my computer). This script filters NaN-values and other onzin-waardes from the CSV file to make it smaller. It also makes a smaller version of the CSV file to test (beacuse then it is a much faster). The cleaned CSV files are then stored in the AIS_DATA_CLEANED folder.
3) Then the cleaned CSV files are loaded into the matrix_creator.py script. This code transforms the CSV file into a Numpy matrix. It creates a test_matrix, train_matrix and val_matrix. The train matrix is also clustered and the same clustering will be applied to the test and val matrices. After clustering the matrices and cluster.joblib files are stored in the matrices folder. In the matrices folder a subfolder is created with the name of the AIS CSV dataset used and also the used random_seed. In that subfolder, the train, test and val matrices are stored. 
4) The script Neural_Network_AIS.py is the main file for creating the neural network. It loads a test, train and val matrix from the matrices folder. Then it trains a NN and when it is finished, it saves the model files to the new_models folder. When a model has None in its name, it means that the model is trained for all clusters. 
5) (Optional) After a model is saved, the directory_celaner.py creates a subfolder in the new_models folder for each model 
