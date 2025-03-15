# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:07:37 2025

@author: evert
"""

import os
import json
import numpy as np
import pandas as pd


import json
import os


# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file
json_file_path = os.path.join(script_dir, "raw_data_rotterdam", "AIS_data_compressed", "AIS_compressed.json")

# Load the JSON file
with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)


