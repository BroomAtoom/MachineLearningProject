import csv
from collections import Counter

# Define the file path
file_path = 'NOAA_AIS/AIS_2020_01_01_fullycleaned.csv'

# Initialize a Counter to keep track of status occurrences
status_counter = Counter()

# Open the CSV file and read the data
with open(file_path, mode='r', newline='') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        # Count the occurrences of each status
        status = row['Status']
        status_counter[status] += 1

# Print the count of each status
for status, count in status_counter.items():
    print(f'Status {status}: {count} occurrences')
