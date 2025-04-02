import os
import pandas as pd

def main():
    filename = "AIS_2020_01_09.csv"
    output_filename = "AIS_2020_01_09_fullycleaned.csv"
    input_folder_name = "COAST_NOAA_AIS_DATA"
    output_folder_name = "AIS_DATA_CLEANED"
    file_path = os.path.join(input_folder_name,filename)
    output_path = os.path.join(output_folder_name, output_filename)
    preprocess_AIS(file_path, output_path)
    df = select_head(output_path, 0.5)
    print(df)


### SOG is speed over ground. Dit moet ook toegevoegd worden. Ook nog toevoegen dat alle schepen die minder dan 100 datapunten hebben.
## COG,Heading,VesselName,IMO,CallSign,VesselType,Length,Width,Draft,Cargo,TransceiverClass
def preprocess_AIS(file_path, output_path):
    print('reading CSV...')
    df = pd.read_csv(file_path)
    
    # Drop all columns with data that are not used. 
    print("Dropping unused columns...")
    df.drop(columns=['COG','Heading','VesselName','IMO','CallSign','VesselType','Length','Width','Draft','Cargo','TransceiverClass'], inplace=True)
    
    # Convert 'Status' column to numeric, forcing errors to NaN for non-numeric values
    print("Converting status column to numeric...")
    df["Status"] = pd.to_numeric(df["Status"], errors="coerce")
    
    # Drop all rows where longitude, latitude, or status are NaN.
    print("Dropping rows with NaN values in LAT, LON, SOG, or Status...")
    df.dropna(axis=0, subset=["LAT", "LON", "Status", "SOG"], inplace=True)
    
    # Filter rows where Status is between 1 and 8
    print("Filtering rows with Status between 0 and 8...")
    df = df[df["Status"].between(0, 8, inclusive="both")]

        
    # Count number of datapoints per MMSI
    print("Counting n0 datapoints")
    mmsi_counts = df['MMSI'].value_counts().reset_index()
    mmsi_counts.columns = ['MMSI', 'DataPoints']
    mmsi_counts = mmsi_counts[mmsi_counts['DataPoints'] > 100]

    # Remove rows in df if mmsi not in mmsi_count:
    print("removing rows")
    df = df[df['MMSI'].isin(mmsi_counts['MMSI'])]


    # Save the cleaned dataframe as CSV
    print("Saving CSV in progres...")
    df.to_csv(output_path, index=False)
    print(f"CSV saved successfully to {output_path}!")


def select_head(csv_file, percentage):
    """ 
    This function will take a .csv and a number (percentage) between 1 and 100.
    If the percentage is 100, nothing changes.
    When the percentage is below 100, it extracts that percentage of rows, 
    and outputs a csv file.
    
    Args:
    - csv_file (str): The path to the input CSV file.
    - percentage (int): The percentage of rows to sample (between 1 and 100).
    
    Returns:
    - A DataFrame containing the sampled rows.
    """
    # Read the CSV file into a DataFrame
    print("Converting csv to df")
    df = pd.read_csv(csv_file)
    
    # Check if the percentage is 100, return the original DataFrame
    if percentage == 100:
        print("Returning the entire dataset.")
        return df

    # Calculate the number of rows to take (top percentage)
    num_rows = int(len(df) * (percentage / 100))

    # Select the top 'num_rows' rows
    df_selected = df.head(num_rows)

    # Save the selected DataFrame to a new CSV file
    output_file = csv_file.replace(".csv", f"_top_{percentage}.csv")
    df_selected.to_csv(output_file, index=False)

    print(f"Extracted the top {percentage}% of rows. Output saved to {output_file}.")
    
    return df_selected

if __name__ == '__main__':
    main()



