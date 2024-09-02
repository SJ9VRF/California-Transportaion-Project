# PlaneNearestPointsFitter

## Description

**PlaneNearestPointsFitter** is a Python class designed to work with 3D point clouds and plane equations. The main functionality of the class is to identify the k-nearest points in a 3D point cloud to a given plane, and then fit a line through these points that best represents their distribution in 3D space.

## Features

- **k-Nearest Points Calculation**: The algorithm calculates the distance of each point in the cloud to the plane and identifies the k points that are closest to the plane.
- **Best-Fit Line Using PCA**: The algorithm fits a line to the k-nearest points using Principal Component Analysis (PCA). This line is the best fit to the points, capturing the direction of maximum variance.
- **Interactive Visualization**: The class generates an interactive 3D plot using Plotly, allowing you to visualize the point cloud, plane, k-nearest points, and the fitted line.

## How It Works

1. **Initialization**: 
   - You initialize the `PlaneNearestPointsFitter` class with a plane equation and a point cloud. The plane equation should be in the format `"ax + by + cz + d = 0"`, where `a`, `b`, `c`, and `d` are the coefficients.
   - The point cloud should be provided as a NumPy array with shape `(n_points, 3)`.

2. **Finding k-Nearest Points**:
   - The `find_k_nearest_points` method calculates the distance of each point in the point cloud to the plane and selects the k-nearest points based on these distances.

3. **Fitting a Line**:
   - The `fit_line_to_points` method applies PCA to the k-nearest points to determine the direction of the best-fit line. This line is not constrained to lie within the plane but rather fits the spatial distribution of the nearest points.

4. **Visualization**:
   - The `visualize` method creates an interactive 3D plot that shows the point cloud, the plane, the k-nearest points, and the best-fit line. This visualization allows for an in-depth exploration of how the points, plane, and line relate to each other in 3D space.

5. **Sample Usage**:
   - Check the sample usage part.
     
## Sample Visualization
- Check interactive visualization in ```sample_run```
<img width="865" alt="Screenshot 2024-09-02 at 1 29 55 PM" src="https://github.com/user-attachments/assets/5851ea68-e4e8-4b49-9ffc-edbb47405f4d">
