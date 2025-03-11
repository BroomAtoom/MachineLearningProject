import json
import os
# from datetime import datetime as dt

# Define the correct file path
file_path = os.path.join("sample_raw_data_rotterdam", "raw_ais_data_2021_rotterdam_1609459200.0_1609545600.0.json")
output_path = os.path.join("sample_raw_data_rotterdam", "pretty_raw_ais_data_2021_rotterdam_1609459200.0_1609545600.0.json")
# Open the JSON file correctly
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    data = data["data"]

# with open(output_path, "w", encoding="utf-8") as file:
#     json.dump(data, file, indent=4)

# for i in range(len(data)):
#     if data[i]["navigation"]["heading"] != None:
#         print(f"{data[i]["navigation"]["status"]}{data[i]["vessel"]["name"]}{data[i]["navigation"]["time"]} heading = {data[i]["navigation"]["heading"]}, course = {data[i]["navigation"]["course"]}")

vessels = []
destinations = []
for i in range(len(data)):
    if data[i]["vessel"]["name"] not in vessels:
        vessels.append(data[i]["vessel"]["name"])
        destinations.append(data[i]["navigation"]["destination"]["name"])

for destination in destinations:
    print(destination)