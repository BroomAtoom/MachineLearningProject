import pandas as pd
import folium
import os

source_directory = r'C:\Users\jarri\Desktop\Spyder_Conda\bigproject\vessel_csvs'
output_directory = r'C:\Users\jarri\Desktop\Spyder_Conda\bigproject\vessel_routes'

for filename in os.listdir(source_directory):
    file_path = os.path.join(source_directory, filename)
    save_path = os.path.join(output_directory, filename.replace('.csv', '.html'))

    # Load the data
    df = pd.read_csv(file_path)
    #df = df[df['status'] != 'moored']


    # Ensure longitude and latitude are numeric
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

    # Drop NaN values
    df = df.dropna(subset=['longitude', 'latitude'])
    if df.empty:
        print(f"Skipping {filename} because it has no valid coordinates.")
        continue

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

