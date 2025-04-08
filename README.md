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

De rest van de bestanden zijn onzin (maar verwijder maar niet voor de zekerheid)

Step-by-step:

1) AIS-CSV files are downloaded from #https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/index.html and put in the COAST_NOAA_AIS_DATA folder
2) The AIS-CSV files are loaded into the preprocess_AIS_branchEvert.py script. This script is a copy of Jarijks script (changed some stuff to make work on my computer)
