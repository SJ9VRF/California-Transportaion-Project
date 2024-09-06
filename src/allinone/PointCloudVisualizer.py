import laspy
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os  # For extracting file name without extension


class PointCloudVisualizer:
    def __init__(self, file_path):
        """
        Initializes the PointCloudVisualizer class.

        Args:
            file_path (str): The path to the .las file.
        """
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract file name without extension
        self.point_cloud = None
        self.point_classes = None
        self.between_points = None
        self.complement_of_between_points = None
        self.bottom_part = None

    def load_las_file(self):
        """
        Loads the .las file and extracts the point cloud data and class information.
        """
        # Open the .las file
        las = laspy.read(self.file_path)

        # Extract point coordinates (X, Y, Z)
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # Extract point classes
        point_classes = las.classification

        # Store the points and classes
        self.point_cloud = points
        self.point_classes = point_classes

    def get_color_map(self):
        """
        Generates a colormap for each point class.

        Returns:
            dict: A dictionary with colors for each class.
        """
        # Unique classes
        unique_classes = np.unique(self.point_classes)

        # Generate a colormap (using matplotlib's 'tab20' for distinct colors)
        cmap = plt.get_cmap('tab20', len(unique_classes))

        # Create a dictionary to map each class to a color
        class_color_map = {cls: cmap(i)[:3] for i, cls in enumerate(unique_classes)}

        return class_color_map

    def train_svm(self, class_a, class_b, points, labels):
        """
        Trains a linear SVM to separate two classes and returns the plane parameters.

        Args:
            class_a, class_b (int): The class labels for the two classes to be separated.
            points (np.ndarray): The point cloud data to train the SVM.
            labels (np.ndarray): The corresponding class labels for the points.

        Returns:
            tuple: Coefficients (a, b, c, d) for the separating plane.
        """
        # Extract points belonging to Class A and Class B
        mask = np.isin(labels, [class_a, class_b])
        points_filtered = points[mask]
        labels_filtered = labels[mask]

        # Train a linear SVM classifier
        svm = SVC(kernel='linear')
        svm.fit(points_filtered, labels_filtered)

        # Get the separating hyperplane parameters
        # The normal vector to the plane is given by the coefficients of the SVM
        w = svm.coef_[0]
        a, b, c = w  # Coefficients for x, y, and z

        # The intercept gives us the constant d in the plane equation
        d = svm.intercept_[0]

        return a, b, c, d

    def calculate_signed_distance(self, point, a, b, c, d):
        """
        Calculates the signed distance of a point from a plane defined by ax + by + cz + d = 0.

        Args:
            point (np.ndarray): A 3D point [x, y, z].
            a, b, c, d (float): Coefficients of the plane equation.

        Returns:
            float: The signed distance from the point to the plane.
        """
        x, y, z = point
        return (a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

    def calculate_between_and_complement_points(self, a1, b1, c1, d1, a2, b2, c2, d2):
        """
        Calculates the points that lie between two planes and their complement, keeping the classification.

        Args:
            a1, b1, c1, d1 (float): Coefficients of Plane 1.
            a2, b2, c2, d2 (float): Coefficients of Plane 2.
        """
        between_points = []
        complement_of_between_points = []

        for i, point in enumerate(self.point_cloud):
            # Calculate signed distances from both planes
            distance_to_plane_1 = self.calculate_signed_distance(point, a1, b1, c1, d1)
            distance_to_plane_2 = self.calculate_signed_distance(point, a2, b2, c2, d2)

            # Check if the point lies between the planes (opposite signs)
            if distance_to_plane_1 * distance_to_plane_2 < 0:
                between_points.append((point, self.point_classes[i]))
            else:
                complement_of_between_points.append((point, self.point_classes[i]))

        # Store the points between the two planes and their complement with classification
        self.between_points = np.array(between_points, dtype=object)
        self.complement_of_between_points = np.array(complement_of_between_points, dtype=object)

    def calculate_bottom_part(self):
        """
        Calculates the bottom part of points from complement_of_between_points with z smaller than avg_z.
        """
        if self.complement_of_between_points is None:
            raise ValueError("No complement_of_between_points calculated. Please run `visualize()` first.")

        # Filter points with class 1 in complement_of_between_points
        points_class_1 = np.array([point[0] for point in self.complement_of_between_points if point[1] == 1])

        if len(points_class_1) == 0:
            raise ValueError("No points with classification label 1 found in complement_of_between_points.")

        # Calculate max and min z values for class 1
        max_z = np.max(points_class_1[:, 2])
        min_z = np.min(points_class_1[:, 2])

        # Calculate avg_z
        avg_z = (max_z + min_z) / 2

        # Set bottom_part as all points in complement_of_between_points with z < avg_z
        self.bottom_part = np.array([point for point in self.complement_of_between_points if point[0][2] < avg_z],
                                    dtype=object)

    def plot_plane(self, a, b, c, d, x_range, y_range):
        """
        Plots a plane using the equation ax + by + cz + d = 0.

        Args:
            a, b, c, d (float): Coefficients of the plane equation.
            x_range (tuple): Range of x values for the plane.
            y_range (tuple): Range of y values for the plane.
        """
        # Create a meshgrid for the plane
        xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 10), np.linspace(y_range[0], y_range[1], 10))

        # Calculate the corresponding z values from the plane equation: z = (-d - ax - by) / c
        zz = (-d - a * xx - b * yy) / c

        # Return the meshgrid points for plotting
        return xx, yy, zz

    def visualize(self):
        """
        Visualizes the point cloud in an interactive mode using Plotly with color coding for point classes.
        Adds two SVM separating planes:
            - Plane 1: Separating Class 1 and Class 2
            - Plane 2: Separating Class 1 and Class 3
        """
        if self.point_cloud is None or self.point_classes is None:
            raise ValueError("No point cloud loaded. Please run `load_las_file()` first.")

        # Get color map for the point cloud based on classes
        class_color_map = self.get_color_map()

        # Create a Plotly figure
        fig = go.Figure()

        # Plot each class separately to have separate legend entries
        unique_classes = np.unique(self.point_classes)
        for cls in unique_classes:
            # Get points of the current class
            indices = np.where(self.point_classes == cls)
            points_class = self.point_cloud[indices]

            # Get color for the current class
            color_rgb = class_color_map[cls]
            color_hex = 'rgb({},{},{})'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255),
                                               int(color_rgb[2] * 255))

            # Add a 3D scatter plot trace for the current class
            fig.add_trace(go.Scatter3d(
                x=points_class[:, 0],
                y=points_class[:, 1],
                z=points_class[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_hex,
                    opacity=0.8
                ),
                name=f"Class {cls}"  # Legend entry for the class
            ))

        # Get the ranges of x and y from the point cloud
        x_range = (np.min(self.point_cloud[:, 0]), np.max(self.point_cloud[:, 0]))
        y_range = (np.min(self.point_cloud[:, 1]), np.max(self.point_cloud[:, 1]))

        # Train SVM and get plane parameters for Class 1 and Class 2 (Plane 1)
        a1, b1, c1, d1 = self.train_svm(1, 2, self.point_cloud, self.point_classes)
        # Get the plane meshgrid points
        xx1, yy1, zz1 = self.plot_plane(a1, b1, c1, d1, x_range, y_range)
        # Add the SVM plane for Class 1 and Class 2 as a surface to the plot
        fig.add_trace(go.Surface(
            x=xx1,
            y=yy1,
            z=zz1,
            opacity=0.5,
            colorscale='Viridis',
            showscale=False,
            name='Plane 1: Class 1 vs Class 2'
        ))

        # Train SVM and get plane parameters for Class 1 and Class 3 (Plane 2)
        a2, b2, c2, d2 = self.train_svm(1, 3, self.point_cloud, self.point_classes)
        # Get the plane meshgrid points
        xx2, yy2, zz2 = self.plot_plane(a2, b2, c2, d2, x_range, y_range)
        # Add the SVM plane for Class 1 and Class 3 as a surface to the plot
        fig.add_trace(go.Surface(
            x=xx2,
            y=yy2,
            z=zz2,
            opacity=0.5,
            colorscale='Cividis',
            showscale=False,
            name='Plane 2: Class 1 vs Class 3'
        ))

        # Calculate points between the two planes and their complement
        self.calculate_between_and_complement_points(a1, b1, c1, d1, a2, b2, c2, d2)

        # Update layout for a better 3D view
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=0),  # No margin
            showlegend=True  # Show the legend
        )

        # Show the plot
        pio.show(fig)

    def visualize_points(self, points, title):
        """
        Visualizes a given set of points with their classifications.
        Args:
            points: A list of tuples where each tuple contains the point coordinates and its classification.
            title: The title for the plot.
        """
        if points is None or len(points) == 0:
            raise ValueError(f"No {title} points calculated.")

        # Create a Plotly figure
        fig = go.Figure()

        # Get color map for the point cloud based on classes
        class_color_map = self.get_color_map()

        # Visualize each class in the points
        for cls in np.unique(self.point_classes):
            class_points = [point for point in points if point[1] == cls]

            if len(class_points) > 0:
                # Extract the points for this class
                points_coords = np.array([p[0] for p in class_points])

                # Get color for the current class
                color_rgb = class_color_map[cls]
                color_hex = 'rgb({},{},{})'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255),
                                                   int(color_rgb[2] * 255))

                # Add the class points to the plot
                fig.add_trace(go.Scatter3d(
                    x=points_coords[:, 0],
                    y=points_coords[:, 1],
                    z=points_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=color_hex,
                        opacity=0.7
                    ),
                    name=f'Class {cls}'
                ))

        # Update layout for a better 3D view
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=0),  # No margin
            showlegend=True  # Show the legend
        )

        # Show the plot
        pio.show(fig)

    def visualize_bottom_part(self):
        """
        Visualizes the bottom part of points from complement_of_between_points with z smaller than avg_z.
        """
        self.visualize_points(self.bottom_part, "Bottom Part of Points (Z < avg_z)")

    def calculate_and_visualize_plane_3(self):
        """
        Calculates and visualizes the SVM separator plane (Plane 3) for class 0 and class 1 in bottom_part.
        Writes the plane coefficients to a .npy file.
        """
        if self.bottom_part is None:
            raise ValueError("Bottom part not calculated. Please run `calculate_bottom_part()` first.")

        # Extract the points and labels for class 0 and class 1 in bottom_part
        points_bottom = np.array([p[0] for p in self.bottom_part])
        labels_bottom = np.array([p[1] for p in self.bottom_part])

        # Train SVM and get plane parameters for Class 0 and Class 1 in bottom_part
        a3, b3, c3, d3 = self.train_svm(0, 1, points_bottom, labels_bottom)

        # Print the coefficients of Plane 3
        print(f"Plane 3 (Class 0 vs Class 1 in Bottom Part): a={a3}, b={b3}, c={c3}, d={d3}")

        # Save the plane coefficients to a .npy file
        plane_coefficients = np.array([a3, b3, c3, d3])
        file_name = f"{self.file_name}_bottomplane.npy"
        npy_path = ''
        np.save(npy_path + file_name, plane_coefficients)
        print(f"Plane 3 coefficients saved to {file_name}")

        # Visualize Plane 3
        x_range = (np.min(points_bottom[:, 0]), np.max(points_bottom[:, 0]))
        y_range = (np.min(points_bottom[:, 1]), np.max(points_bottom[:, 1]))
        xx3, yy3, zz3 = self.plot_plane(a3, b3, c3, d3, x_range, y_range)

        # Create a Plotly figure for Plane 3
        fig = go.Figure()

        # Plot the entire point cloud
        class_color_map = self.get_color_map()
        for cls in np.unique(self.point_classes):
            indices = np.where(self.point_classes == cls)
            points_class = self.point_cloud[indices]
            color_rgb = class_color_map[cls]
            color_hex = 'rgb({},{},{})'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255),
                                               int(color_rgb[2] * 255))

            fig.add_trace(go.Scatter3d(
                x=points_class[:, 0],
                y=points_class[:, 1],
                z=points_class[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_hex,
                    opacity=0.8
                ),
                name=f"Class {cls}"
            ))

        # Add Plane 3 to the plot
        fig.add_trace(go.Surface(
            x=xx3,
            y=yy3,
            z=zz3,
            opacity=0.5,
            colorscale='Blues',
            showscale=False,
            name='Plane 3: Class 0 vs Class 1 in Bottom Part'
        ))

        # Update layout for better 3D view
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True
        )

        # Show the plot
        pio.show(fig)


# Example Usage:
if __name__ == "__main__":
    # Code to execute when the script is run directly

    las_file_path = ""

    visualizer = PointCloudVisualizer(las_file_path)

    # Load the .las file
    visualizer.load_las_file()

    # Visualize the point cloud with two SVM planes (Class 1 vs Class 2 and Class 1 vs Class 3)
    visualizer.visualize()

    # Calculate and visualize the bottom part
    visualizer.calculate_bottom_part()
    visualizer.visualize_bottom_part()

    # Calculate and visualize Plane 3 (Class 0 vs Class 1 in bottom_part) and save the plane to a .npy file
    visualizer.calculate_and_visualize_plane_3()
