# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:13:01 2025

@author: evert
"""

import os
import joblib

# Model path
model_path = r"C:\Users\evert\Documents\TU-Delft\TIL Master\ME44312 Machine Learning for Transport and Multi-Machine Systems\Old_NN_models\JSON_AIS_first_model_accuracy_79.51%.joblib"

# Load the model
model = joblib.load(model_path)

# Extract and print the filename
filename = os.path.basename(model_path)
print("Model filename:", filename)

input_data = [42.4,19.4,12]

prediction = model.predict(input_data)
print("Predicted output:", prediction)

