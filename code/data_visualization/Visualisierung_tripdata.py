import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from shapely.wkt import loads
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

def visualize_trip_data():
    path = "store/baseline/baseline_1"
    # Load the data from the CSV file
    file_path = f'{path}/tripdata2015-07-06.csv'
    data = pd.read_csv(file_path)
    # Convert 'total_seconds' to 'hour_of_day' for easier analysis
    data['hour_of_day'] = pd.to_datetime(data['total_seconds'], unit='s').dt.hour
    # Group the data by hour and count the number of trips in each hour
    hourly_trip_data = data.groupby('hour_of_day').size().reset_index(name='trip_count')







    # fig 1: number of trips per hour
    # This chart will visually represent the number of trips per hour
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour_of_day', y='trip_count', data=hourly_trip_data,color='#40A0A0')
    sns.lineplot(x='hour_of_day', y='trip_count', data=hourly_trip_data, marker='o',color= 'red')
    plt.title('Number of Trips per Hour',fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day',fontsize=12, fontweight='bold')
    plt.ylabel('Number of Trips',fontsize=12, fontweight='bold')
    plt.xticks(range(0, 24)) # Setting x-axis ticks for each hour
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.show()
    plt.savefig(f"{path}/trips_per_hour.png")







    # fig 2:  pickup->dropoff distance    vs.   ÖPNV distance
    # Load CSV files
    trip_data_file_path = 'code/data_output/tripdata2015-07-06.csv'
    taxi_zones_file_path = 'code/data/taxi_zones.csv'
    trip_data = pd.read_csv(trip_data_file_path)
    taxi_zones = pd.read_csv(taxi_zones_file_path)
    # Convert polygon strings to shapely polygon objects
    taxi_zones['polygon'] = taxi_zones['the_geom'].apply(loads)
    # Calculate the latitude and longitude of the center for each zone
    taxi_zones['center'] = taxi_zones['polygon'].apply(lambda p: (p.centroid.y, p.centroid.x))
    # Create a dictionary to map zone IDs to their center latitude and longitude
    zone_centers = dict(zip(taxi_zones['LocationID'], taxi_zones['center']))
    # Define a function to calculate the distance between two zones in kilometers
    def calculate_distance_km(zone1_id, zone2_id):
        if zone1_id in zone_centers and zone2_id in zone_centers:
            # Use geodesic from geopy to calculate the distance
            return geodesic(zone_centers[zone1_id], zone_centers[zone2_id]).kilometers
        else:
            return np.nan
    # Calculate distances for each trip in the dataset
    trip_data['pickup_to_dropoff_distance_km'] = trip_data.apply(lambda x: calculate_distance_km(x['passenger_pickup_zone_id'], x['passenger_dropoff_zone_id']), axis=1)
    trip_data['dropoff_to_destination_distance_km'] = trip_data.apply(lambda x: calculate_distance_km(x['passenger_dropoff_zone_id'], x['destination_id']), axis=1)
    # Convert total_seconds to hours for aggregation
    trip_data['hour_of_day'] = pd.to_datetime(trip_data['total_seconds'], unit='s').dt.hour
    # Calculate the average distance for all trips in each hour
    average_distances_per_hour_km = trip_data.groupby('hour_of_day').agg({
        'pickup_to_dropoff_distance_km': 'mean',
        'dropoff_to_destination_distance_km': 'mean'
    }).reset_index()
    # Plotting the bar charts
    dark_blue = 'navy'  # Color for the first bar chart
    light_blue = 'skyblue'  # Color for the second bar chart
    black = '#000033'
    # #simple Plot:  Average distance from pickup to dropoff per hour
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='hour_of_day', y='pickup_to_dropoff_distance_km', data=average_distances_per_hour_km, color=dark_blue)
    # plt.title('Average Distance from Pickup to Dropoff per Hour (km)')
    # plt.xlabel('Hour of Day')
    # plt.ylabel('Average Distance (km)')
    # plt.show()

    # #simple Plot: Average distance from pickup to destination per hour
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='hour_of_day', y='dropoff_to_destination_distance_km', data=average_distances_per_hour_km, color=light_blue)
    # plt.title('Average Distance from Pickup to Destination (km)')
    # plt.xlabel('Hour of Day')
    # plt.ylabel('Average Distance (km)')
    # plt.show()


    # Combined plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour_of_day', y='pickup_to_dropoff_distance_km', data=average_distances_per_hour_km, color=dark_blue, label='Pickup to Dropoff')
    sns.barplot(x='hour_of_day', y='dropoff_to_destination_distance_km', data=average_distances_per_hour_km, color=light_blue, alpha=0.6, label='Dropoff to Destination', bottom=average_distances_per_hour_km['pickup_to_dropoff_distance_km'])
    plt.title('Comparison of Average Distances (km)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Distance (km)')
    plt.legend()
    plt.show()







    #fig 3: driver pickup distance
    # calculate distance from driver_start_zone_id to passenger_pickup_zone_id
    trip_data['driver_to_pickup_distance_km'] = trip_data.apply(
        lambda x: calculate_distance_km(x['driver_start_zone_id'], x['passenger_pickup_zone_id']), axis=1
    )
    # average distance from driver_start_zone_id to passenger_pickup_zone_id（km)
    average_driver_to_pickup_per_hour = trip_data.groupby('hour_of_day').agg({
        'driver_to_pickup_distance_km': 'mean'
    }).reset_index()
    # plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour_of_day', y='driver_to_pickup_distance_km', data=average_driver_to_pickup_per_hour, color=dark_blue)
    sns.lineplot(x='hour_of_day', y='driver_to_pickup_distance_km', data=average_driver_to_pickup_per_hour, marker = 'o',color='red')
    plt.title('Average Distance for Driver to Pickup Passenger (km)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Distance (km)')
    plt.ylim(0.5, 1.8) 
    plt.show()








    # fig 4: compare all the distances in the whole trip 
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour_of_day', y='driver_to_pickup_distance_km', data=average_driver_to_pickup_per_hour, color=black, label='Start to Pickup')
    sns.barplot(x='hour_of_day', y='pickup_to_dropoff_distance_km', data=average_distances_per_hour_km, color=dark_blue,label='Pickup to Dropoff',bottom=average_driver_to_pickup_per_hour['driver_to_pickup_distance_km'])
    sns.barplot(x='hour_of_day', y='dropoff_to_destination_distance_km', data=average_distances_per_hour_km, color=light_blue,  label='Dropoff to Destination', bottom=average_distances_per_hour_km['pickup_to_dropoff_distance_km']+average_driver_to_pickup_per_hour['driver_to_pickup_distance_km'])
    plt.title('Comparison of Average Distances (Whole Trip) (km)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Distance (km)')
    plt.legend()
    plt.show()








    # fig 5: average_time_reduction 
    # calculate average_time_reduction per hour
    trip_data['time_reduction'] = trip_data['time_reduction'] / 60 
    average_time_reduction_per_hour = trip_data.groupby('hour_of_day').agg({
        'time_reduction': 'mean'
    }).reset_index()
    # plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour_of_day', y='time_reduction', data=average_time_reduction_per_hour, color='navy')
    sns.lineplot(x='hour_of_day', y='time_reduction', data=average_time_reduction_per_hour, marker='o',color='red')
    plt.title('Average Time Reduction per Order per Hour')
    plt.xlabel('Hour of Day')
    plt.ylim(30,70)
    plt.ylabel('Average Time Reduction (minutes)')
    plt.show()







    # fig 6: combi_route percentage 
    trip_data['hour_of_day'] = pd.to_datetime(trip_data['total_seconds'], unit='s').dt.hour

    hourly_combi_count = trip_data.groupby(['hour_of_day', 'combi_route']).size().unstack(fill_value=0)

    hourly_combi_count['total'] = hourly_combi_count[True] + hourly_combi_count[False]

    hourly_combi_count['true_percentage'] = (hourly_combi_count[True] / hourly_combi_count['total']) * 100
    hourly_combi_count['false_percentage'] = (hourly_combi_count[False] / hourly_combi_count['total']) * 100

    plt.figure(figsize=(12, 6))
    plt.barh(hourly_combi_count.index, hourly_combi_count['true_percentage'], label='Combi Trip', color='navy')
    plt.barh(hourly_combi_count.index, hourly_combi_count['false_percentage'], left=hourly_combi_count['true_percentage'], label='Direct Trip', color='teal')
    plt.title('Combi Trip vs Direct Trip (Proportion)')
    plt.ylabel('Hour of Day')
    plt.xlabel('Percentage')
    plt.yticks(range(24), [f"{hour}" for hour in range(24)])
    plt.legend()
    plt.show()

    # Start- und Enddatum für die Dateinamen
    start_date = datetime.date(2015, 7, 6)
    end_date = datetime.date(2015, 7, 31)

    # Erstellung einer Liste von Datumsangaben im angegebenen Bereich
    date_range = pd.date_range(start_date, end_date)

    # Liste für die Ergebnisse
    daily_results = []

    # Durchlaufen aller Dateien
    for single_date in date_range:
        file_date = single_date.strftime("%Y-%m-%d")
        file_path = f'code/data_output/tripdata{file_date}.csv'
        try:
            trip_data = pd.read_csv(file_path)
            # Berechnung der Zeitreduktion in Minuten
            trip_data['time_reduction'] = trip_data['time_reduction'] / 60
            # Berechnung des durchschnittlichen Zeitreduktion pro Tag
            mean_time_reduction = trip_data['time_reduction'].mean()
            # Hinzufügen des Ergebnisses zur Liste
            daily_results.append({'date': file_date, 'average_time_reduction': mean_time_reduction})
        except FileNotFoundError:
            print(f"Datei {file_path} nicht gefunden")

    # Erstellung eines DataFrame aus den Ergebnissen
    results_df = pd.DataFrame(daily_results)

    # Erstellung des Plots
    plt.figure(figsize=(12, 6))
    sns.barplot(x='date', y='average_time_reduction', data=results_df, color='navy')
    plt.xticks(rotation=45)
    plt.title('Durchschnittliche Zeitreduktion pro Tag')
    plt.xlabel('Datum')
    plt.ylabel('Durchschnittliche Zeitreduktion (Minuten)')
    plt.show()


    ##################

    # Start- und Enddatum für die Dateinamen
    start_date = datetime.date(2015, 7, 6)
    end_date = datetime.date(2015, 7, 31)

    # Erstellung einer Liste von Datumsangaben im angegebenen Bereich
    date_range = pd.date_range(start_date, end_date)

    # Liste für die Ergebnisse
    daily_status_counts = []

    # Durchlaufen aller Dateien
    for single_date in date_range:
        file_date = single_date.strftime("%Y-%m-%d")
        file_path = f'code/data_output/driverdata{file_date}.csv'
        try:
            driver_data = pd.read_csv(file_path)
            # Zählen der Häufigkeit jedes Status
            status_counts = driver_data['status'].value_counts().reset_index()
            status_counts.columns = ['status', 'count']
            status_counts['date'] = file_date
            # Hinzufügen des Ergebnisses zur Liste
            daily_status_counts.extend(status_counts.to_dict('records'))
        except FileNotFoundError:
            print(f"Datei {file_path} nicht gefunden")

    # Erstellung eines DataFrame aus den Ergebnissen
    status_counts_df = pd.DataFrame(daily_status_counts)

    # Erstellung des Plots
    plt.figure(figsize=(12, 6))
    sns.barplot(x='date', y='count', hue='status', data=status_counts_df)
    plt.xticks(rotation=45)
    plt.title('Verteilung des Status pro Tag')
    plt.xlabel('Datum')
    plt.ylabel('Anzahl der Vorkommen')
    plt.legend(title='Status')
    plt.show()