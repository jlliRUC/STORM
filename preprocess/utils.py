import pickle
import numpy as np
import json


def boundary(df):
    """
    For those datasets that not are not located in one city, we calculate the boundary of longitude and latitude.
    :return:
    """
    min_lon = float("inf")
    max_lon = -float("inf")
    min_lat = float("inf")
    max_lat = -float("inf")

    for i in range(df.shape[0]):
        traj = df.loc[i, "Locations"]
        for point in traj:
            if point[0] > max_lon:
                max_lon = point[0]
            elif point[0] < min_lon:
                min_lon = point[0]
            if point[1] > max_lat:
                max_lat = point[1]
            elif point[1] < min_lat:
                min_lat = point[1]

    print(f"min_lon: {min_lon}, max_lon: {max_lon}, min_lat: {min_lat}, max_lat: {max_lat}")

    return min_lon, min_lat, max_lon, max_lat


def point_in_region(region_settings, lon, lat):
    if lon <= region_settings['min_lon'] or lon >= region_settings['max_lon'] \
            or lat <= region_settings['min_lat'] or lat >= region_settings['max_lat']:
        return False
    return True


def filter_inregion(df, region_settings):
    # Region filter  # For cell pretraining, all points need to be within the region
    df["inregion"] = df["Locations"].map(lambda traj: sum([point_in_region(region_settings, point[0], point[1]) for point in traj]) == len(traj))
    return df[df.inregion == True]


def length_of_traj(df_location):
    len_list = []
    for traj in df_location:
        len_list.append(len(traj))
    return len_list


def mean_time_interval(df):
    timestamps_list = df['Timestamps'].values
    sum_interval = 0
    num_interval = 0
    for timestamps in timestamps_list:
        if not isinstance(timestamps, list):
            timestamps = json.loads(timestamps)
            sum_temp = 0
            for i in range(1, len(timestamps)):
                sum_temp += (timestamps[i] - timestamps[i - 1])
            sum_interval += sum_temp
            num_interval += len(timestamps) - 1
    mean_interval = sum_interval / num_interval
    return mean_interval


def traj_stats(df):
    # Temporal information
    mean_interval = mean_time_interval(df)

    # Spatial information
    len_list = length_of_traj(df.Locations)
    data = np.array(len_list)
    print(f"We have {df.shape[0]} trajectories")
    print(f"We have {data.sum()} points")
    print(f"mean_length is {round(np.mean(data), 4)}")
    print(f"mean_time_interval is {round(mean_interval, 4)}")
    print(f"scale_point >= 30 is {data[data >= 30].sum()}")
    print(f"min_length is {np.min(data)}")
    print(f"max_length is {np.max(data)}")
    print(f"length>=40000 is {data[data >= 40000].shape[0]}")
    print(f"median length is {np.median(data)}")
    print(f"90% percentile is {np.percentile(data, 90, axis=0)}")
    print(f"95% percentile is {np.percentile(data, 95, axis=0)}")
    print(f"length<30 is {data[data < 30].shape[0]}, {data[data < 30].shape[0] / data.shape[0]:.2%}")
    print(f"length>=30 is {data[data >= 30].shape[0]}, mean_length is {round(np.mean(data[data >= 30]), 4)}")
    length1 = data.shape[0] - data[data < 30].shape[0] - data[data > 4000].shape[0]
    print(f"30<=length<=2000 is {length1}, {length1 / data.shape[0]:.2%}")
    length2 = data.shape[0] - data[data < 30].shape[0] - data[data > 100].shape[0]
    print(f"30<=length<=100 is {length2}, {length2 / data.shape[0]:.2%}")


def filter_inregion_multi(df, city_name):
    # 01/14/2024, to see the scale of dataset when narrowing the range of the region.

    if city_name == "porto":
        region_settings_list = [{"min_lon": -8.6868, "max_lon": -8.5611, "min_lat": 41.1128, "max_lat": 41.1928},
                                {"min_lon": -8.735152, "max_lon": -8.156309, "min_lat": 40.953673, "max_lat": 41.307945}]

    elif city_name == "beijing":
        region_settings_list = [{"min_lon": 116.0719, "max_lon": 116.7174, "min_lat": 39.6872, "max_lat": 40.2177},
                                {"min_lon": 115.41666, "max_lon": 117.5, "min_lat": 39.45, "max_lat": 41.05}]

    for region_settings in region_settings_list:
        df_filter = filter_inregion(df, region_settings)
        traj_stats(df_filter)


def trajectory_visualization(traj, title):
    """
    Visualize the input trajectory. Each point of trajectory is marked with its order
    :param traj:
    :return:
    """
    import matplotlib.pyplot as plt

    x = traj[:, 0]
    y = traj[:, 1]

    # Plot the trajectory
    plt.plot(x, y, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Coordinator Trajectory Visualization')
    for i in range(traj.shape[0]):
        plt.text(traj[i, 0], traj[i, 1], str(i))
    # Show the plot
    plt.savefig(f"repo/{title}")
    plt.show()


def folium_visualization_traj(traj, title):
    """
    Visualize the input trajectory in real map via folium.
    :param traj:
    :return:
    """
    import folium

    coordinates = [(lat, lon) for lon, lat in traj]
    # Sample coordinates (latitude, longitude)

    # Create a folium map centered at the first coordinate
    mymap = folium.Map(location=coordinates[0], zoom_start=15)

    # Create a polyline using the coordinates
    folium.PolyLine(locations=coordinates, color='blue').add_to(mymap)

    # Save the map to an HTML file
    mymap.save(f"repo/{title}")


def folium_visualization_point(traj, title):
    """
    Visualize the point of input trajectory in real map via folium.
    :param traj:
    :return:
    """
    import folium
    coordinates = [(lat, lon) for lon, lat in traj]
    # Create a folium map centered at a specific location
    map_center = coordinates[0]
    mymap = folium.Map(location=map_center, zoom_start=15)

    # Define the coordinates of the points

    # Add markers for each point
    for coord in coordinates:
        folium.Marker(location=coord, popup=f"Point {coord}").add_to(mymap)

    # Save the map to an HTML file or display it
    mymap.save(f"repo/{title}")


def sorted_set(a, k):
    values = sorted(list(set(a)), reverse=True)[:k]
    result = [a.index(value) for value in values]

    return result


def rle_index(seq):
    result = [0]
    for i in range(1, len(seq)):
        if seq[i] != seq[result[-1]]:
            result.append(i)
    return result


if __name__ == "__main__":
    import pickle
    dataset_name = "porto"
    path = f"../data/{dataset_name}/{dataset_name}.pkl"
    df = pickle.load(open(path, "rb"))
    traj_stats(df)
