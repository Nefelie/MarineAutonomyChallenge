import matplotlib.pyplot as plt
import numpy as np

def plot_search_pattern(coordinates):
    latitudes = [coord[0] for coord in coordinates]
    longitudes = [coord[1] for coord in coordinates]
    
    # Calculate the center of the bounding box
    center_latitude = sum(latitudes) / len(latitudes)
    center_longitude = sum(longitudes) / len(longitudes)
    
    # Calculate the distances from the center to each edge
    distances = [np.sqrt((lat - center_latitude)**2 + (lon - center_longitude)**2) for lat, lon in coordinates]
    max_distance = max(distances)
    
    # Generate spiral coordinates
    spiral_coordinates = []
    num_points = 100  # Adjust this parameter to control the number of points in the spiral
    num_turns = 5  # Adjust this parameter to control the number of turns in the spiral
    
    for i in range(num_points):
        angle = i * (2 * np.pi * num_turns / num_points)
        radius = max_distance * (1 - i / num_points)
        lat = center_latitude + radius * np.sin(angle)
        lon = center_longitude + radius * np.cos(angle)
        spiral_coordinates.append((lat, lon))
    
    fig, ax = plt.subplots()

    # Plot the search pattern
    ax.plot(longitudes, latitudes, marker='o', color='k')
    # Connect the last and first coordinates with a line to close the shape
    ax.plot([longitudes[-1], longitudes[0]], [latitudes[-1], latitudes[0]], color='k')
    # Plot the spiral pattern
    spiral_latitudes = [coord[0] for coord in spiral_coordinates]
    spiral_longitudes = [coord[1] for coord in spiral_coordinates]
    ax.plot(spiral_longitudes, spiral_latitudes, marker='o', color='r')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.show()

# Example usage with the given coordinates
coordinates = [(51.014670, -1.496181), (51.014467, -1.495044), (51.015209, -1.492791), (51.015425, -1.493821)]
plot_search_pattern(coordinates)
