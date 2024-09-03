import open3d as o3d
import laspy
import numpy as np
from collections import Counter
from matplotlib.colors import to_rgb

# Read LAS file
file_paths = ["ChicoADA144_r3_Left_GND_2.las",
              "NAP121_6_Left_gnd_0.las",
              'SF1_19_run3_1_Right_gnd_2_1.las',
              'SUT20_20231019_run1_Right_1000m_GND_1.las',
              'woodlandADA_r8_Left_GND_20.las',
              'woodlandADA_r10_Left_GND_13.las']
main_path = "/Users/arefehyavary/Downloads/pointclouds w updated classification/"
file_paths = [main_path + item for item in file_paths]
file_path = file_paths[0]

def visualize_las_point_cloud(file_path):
    # Load .las file
    las = laspy.read(file_path)

    # Extract the point cloud data
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def get_classification_values(file_path):
    # Load .las file
    las = laspy.read(file_path)

    # Extract the classification attribute
    classifications = las.classification

    return classifications

# Example usage:
classifications = get_classification_values(file_path)

# Print the classifications
print(Counter(classifications))

def get_las_point_cloud_with_classification(file_path, classification_value):
    # Load .las file
    las = laspy.read(file_path)

    # Extract the point cloud data
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Extract the classification attribute
    classifications = las.classification

    # Filter points based on the classification attribute
    filtered_points = points[classifications == classification_value]
    return filtered_points

def visualize_las_point_cloud_with_classification(file_path, classification_value):
    # Load .las file
    las = laspy.read(file_path)

    # Extract the point cloud data
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Extract the classification attribute
    classifications = las.classification

    # Filter points based on the classification attribute
    filtered_points = points[classifications == classification_value]

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def filter_las_point_cloud_with_classification(file_path, classification_value):
    # Load .las file
    las = laspy.read(file_path)

    # Extract the point cloud data
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Extract the classification attribute
    classifications = las.classification

    # Filter points based on the classification attribute
    filtered_points = points[classifications == classification_value]
    return filtered_points

def fit_plane_to_point_cloud(points):
    """
    Fit a plane to a given set of points using RANSAC.

    Parameters:
    points (numpy.ndarray): Array of points with shape (n_points, 3).

    Returns:
    tuple: Plane model coefficients (a, b, c, d) for the plane equation ax + by + cz + d = 0
    """
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Use RANSAC to fit a plane to the point cloud
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)

    [a, b, c, d] = plane_model
    return plane_model, inliers

def create_plane_mesh(plane_model, centroid, size=1.0):
    """
    Create a mesh for the plane given the plane model.

    Parameters:
    plane_model (tuple): Plane model coefficients (a, b, c, d).
    centroid (numpy.ndarray): A point on the plane, typically the centroid of inlier points.
    size (float): Size of the plane mesh.

    Returns:
    o3d.geometry.TriangleMesh: Mesh representation of the plane.
    """
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Create orthogonal vectors to the normal
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Generate the plane mesh points
    plane_points = []
    for dx in np.linspace(-size, size, 10):
        for dy in np.linspace(-size, size, 10):
            point = centroid + dx * u + dy * v
            plane_points.append(point)

    plane_points = np.array(plane_points)

    # Create the mesh for the plane
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(plane_points)

    # Define faces of the plane
    faces = []
    for i in range(9):
        for j in range(9):
            faces.append([i * 10 + j, i * 10 + (j + 1), (i + 1) * 10 + j])
            faces.append([(i + 1) * 10 + j, i * 10 + (j + 1), (i + 1) * 10 + (j + 1)])

    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh

def visualize_point_cloud_with_plane(points, plane_model, inliers):
    """
    Visualize the point cloud and the fitted plane.

    Parameters:
    points (numpy.ndarray): Array of points with shape (n_points, 3).
    plane_model (tuple): Plane model coefficients (a, b, c, d).
    inliers (numpy.ndarray): Indices of inlier points that lie on the plane.
    """
    # Create an Open3D PointCloud object for the input points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color the inlier points
    colors = np.zeros((points.shape[0], 3))
    colors[inliers] = [1, 0, 0]  # Red color for inliers
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Extract the inlier points
    inlier_points = points[inliers]

    # Create the plane mesh
    plane_mesh = create_plane_mesh(plane_model, inlier_points.mean(axis=0))

    # Visualize the point cloud and the plane mesh
    o3d.visualization.draw_geometries([pcd, plane_mesh])

# Example usage
if __name__ == "__main__":
    classification_value = 3
    points = filter_las_point_cloud_with_classification(file_path, classification_value)
    plane_model, inliers = fit_plane_to_point_cloud(points)
    visualize_point_cloud_with_plane(points, plane_model, inliers)
    print(f"Plane model coefficients: {plane_model}")

