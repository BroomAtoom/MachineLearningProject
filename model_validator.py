# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:28:04 2025

@author: evert
"""

import os
import joblib
import  numpy as np

# Dynamic parts
accuracy = 89.68
cluster = None

# Build folder and filename
folder_name = f"model_{accuracy}_cluster_{cluster}"
model_filename = f"CSV_AIS_first_model_accuracy_{accuracy}%.joblib"

# Full path
model_path = os.path.join("new_models", folder_name, model_filename)

# Load model
model = joblib.load(model_path)

print(f"Model loaded from: {model_path}")

