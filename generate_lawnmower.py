import matplotlib.pyplot as plt

def get_orientation(x1, y1, x2, y2, x3, y3, x4, y4):
    """gets the longer axis"""
    lat_dist = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
    lon_dist = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
    return "lat" if lat_dist > lon_dist else "lon"


def generate_divisions(x1, y1, x2, y2, x3, y3, x4, y4, num_strips):
    """divides the longer axis into multiple sub lines"""
    axis = get_orientation(x1, y1, x2, y2, x3, y3, x4, y4)
    if axis == "lat":
        max_coord = max(x1, x2, x3, x4)
        min_coord = min(x1, x2, x3, x4)
    else: # lon
        max_coord = max(y1, y2, y3, y4)
        min_coord = min(y1, y2, y3, y4)
    segment_length = (max_coord - min_coord) / num_strips
    divisions = [min_coord + i * segment_length for i in range(num_strips + 1)]
    return divisions


def generate_waypoints(x1, y1, x2, y2, x3, y3, x4, y4, divisions):
    """generates the required waypoints but unordered"""
    axis = get_orientation(x1, y1, x2, y2, x3, y3, x4, y4)
    waypoints = []
    if axis == "lon":
        for point in divisions:
            waypoints.extend([(point, max(x1, x2, x3, x4)), (point, min(x1, x2, x3, x4))])
    else:  # lat axis
        for point in divisions:
            waypoints.extend([(max(y1, y2, y3, y4), point), (min(y1, y2, y3, y4), point)])
    return waypoints

def sort_waypoints(waypoints):
    axis = get_orientation(x1, y1, x2, y2, x3, y3, x4, y4)
    
    if axis == "lat":
        waypoints.sort(key=lambda p: (p[1], p[0]))
        current_coord = waypoints[0][1]
        get_coord = lambda p: p[1]
        set_coord = lambda p, value: (p[0], value)
    else:
        waypoints.sort(key=lambda p: (p[0], p[1]))
        current_coord = waypoints[0][0]
        get_coord = lambda p: p[0]
        set_coord = lambda p, value: (value, p[1])

    grouped_points = []
    current_group = []

    for point in waypoints:
        if get_coord(point) != current_coord:
            grouped_points.append(current_group)
            current_group = []
            current_coord = get_coord(point)
        current_group.append(point)

    grouped_points.append(current_group)

    for i in range(len(grouped_points)):
        if i % 2 == 1:
            grouped_points[i].reverse()

    return [point for group in grouped_points for point in group]


def waypoints_to_WPL(waypoints, filename):
    # Generate the MMWPL waypoints
    mmwpl_waypoints = []
    for i, waypoint in enumerate(waypoints):
        lat = waypoint[0]
        lon = waypoint[1]
        mmwpl = f"$MMWPL,{lat:.6f},N,{lon:.6f},W,WPT {i+1}"
        mmwpl_waypoints.append(mmwpl)

    # Generate the MMRTE tracks
    mmrte_tracks = []
    current_track = 1
    mmrte_track = f"$MMRTE,2,{current_track},c,TRACK {current_track},"
    for i, waypoint in enumerate(waypoints):
        mmrte_track += f"WPT {i+1},"
        if (i+1) % 5 == 0 and i != len(waypoints)-1:
            mmrte_tracks.append(mmrte_track.rstrip(','))
            current_track += 1
            mmrte_track = f"$MMRTE,2,{current_track},c,TRACK {current_track},"
    mmrte_tracks.append(mmrte_track.rstrip(','))

    # Write to the text file
    with open(filename, 'w') as file:
        for mmwpl in mmwpl_waypoints:
            file.write(mmwpl + '\n')
        for mmrte in mmrte_tracks:
            file.write(mmrte + '\n')


# lat longer
x1, y1 = 00044.755897, 5050.710799
x2, y2 = 00044.755897, 5050.732397
x3, y3 = 00044.704588, 5050.732397
x4, y4 = 00044.704588, 5050.710799

# lon longer
x1, y1 = 00044.755897, 5050.710799
x2, y2 = 00044.755897, 5050.792397
x3, y3 = 00044.704588, 5050.792397
x4, y4 = 00044.704588, 5050.710799
num_strips = 5

waypoints = sort_waypoints(generate_waypoints(x1, y1, x2, y2, x3, y3, x4, y4, generate_divisions(x1, y1, x2, y2, x3, y3, x4, y4, num_strips)))

waypoints_to_WPL(waypoints, "waypoints_lawnmower.txt")

# def plot_lawnmower_path(waypoints):
#     x_values = [point[1] for point in waypoints]  # longitude
#     y_values = [point[0] for point in waypoints]  # latitude

#     plt.plot(x_values, y_values, '-o')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('Lawnmower Path')
#     plt.grid(True)
#     plt.show()

# plot_lawnmower_path(waypoints)