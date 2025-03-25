import pandas as pd
import folium


### Functie om kleinere dataset te samplen. Input df of csv en percentage voor hoeveel van de dataset.
def main():
    file = r'C:\Users\jarri\Desktop\Spyder_Conda\MachineLearningProject\MachineLearningProject\NOAA_AIS\AIS_2020_01_01_cleaned.csv'
    #vessel_counts, df_sorted = extract_unique_vessels(file)
    #print(vessel_counts)
    df = extract_unique_mmsi(file)
    print(df['DataPoints'].to_numpy().mean())

def extract_unique_vessels(csv_file):
    """Extracts all unique vessels from a CSV AIS data file.
    Returns:
    - DataFrame with vessel names and number of data points, sorted by count.
    - DataFrame sorted by vessel name primarily, then timestamp.
    """
    print("creating df")
    df = pd.read_csv(csv_file)
    
    # Handling missing vessel names by filling with 'Unknown'
    print("handling missing values")
    df['VesselName'] = df['VesselName'].fillna('Unknown')
    
    # Count number of occurrences per vessel
    print("counting")
    vessel_counts = df['VesselName'].value_counts().reset_index()
    vessel_counts.columns = ['VesselName', 'DataPoints']
   
    # Sort the full dataset by VesselName first, then by timestamp
    print("sorting")
    df_sorted = df.sort_values(by=['VesselName', 'BaseDateTime'])
    print("all done")
    return vessel_counts, df_sorted

def extract_unique_mmsi(csv_file):
    """Extracts all unique MMSI numbers from a CSV AIS data file.
    Returns:
    - DataFrame with MMSI numbers and number of data points, sorted by count.
    """
    df = pd.read_csv(csv_file)
    
    # Count number of occurrences per MMSI
    mmsi_counts = df['MMSI'].value_counts().reset_index()
    mmsi_counts.columns = ['MMSI', 'DataPoints']
    
    return mmsi_counts


def vesselonmap(df, save_path):
    """This function will take a df with AIS data as input, 
    and create an HTML file with the coordinates from the data plotted
     on a map."""

    # Ensure longitude and latitude are numeric
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

    # Drop NaN values
    df = df.dropna(subset=['longitude', 'latitude'])

    # Get the bounds of the route
    south = df['latitude'].min()
    north = df['latitude'].max()
    west = df['longitude'].min()
    east = df['longitude'].max()

    # Create a Folium map centered around the midpoint of the route
    map_center = [(north + south) / 2, (east + west) / 2]

    # Create a Folium map
    m = folium.Map(location=map_center, zoom_start=12, tiles='CartoDB Positron', control_scale=True)  # Other options: 'OpenStreetMap', 'Stamen Terrain'

    # Add vessel path as a PolyLine
    folium.PolyLine(
        locations=list(zip(df['latitude'], df['longitude'])), 
        color="red", 
        weight=2, 
        opacity=0.8
    ).add_to(m)

    # Add markers for start and end points
    folium.Marker(
        location=[df.iloc[0]['latitude'], df.iloc[0]['longitude']], 
        popup="Start", 
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        location=[df.iloc[-1]['latitude'], df.iloc[-1]['longitude']], 
        popup="End", 
        icon=folium.Icon(color="red")
    ).add_to(m)

    # Set the bounds to fit the entire route
    m.fit_bounds([[south, west], [north, east]])

    m.save(save_path)  # Save as an interactive HTML file

if __name__ == "__main__":
    main()