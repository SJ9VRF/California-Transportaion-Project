import numpy as np
import plotly.graph_objects as go

"""
Short Comment:
# Split the object into top_part and bottom_part based on the middle z value of the middle section
"""

class PointCloudProcessor:
    def __init__(self, num_points=1000):
        """
        Initialize the PointCloudProcessor and generate a random point cloud.
        :param num_points: Number of points to generate.
        """
        self.point_clouds = self.generate_random_point_cloud(num_points)

    def generate_random_point_cloud(self, num_points):
        """
        Generate a random point cloud with x, y, z coordinates and integer labels from 0 to 4.
        :param num_points: Number of points to generate.
        :return: A NumPy array representing the point cloud.
        """
        xyz = np.random.rand(num_points, 3) * 100  # Random x, y, z values
        labels = np.random.randint(0, 5, size=(num_points, 1))  # Random labels between 0 and 4
        return np.hstack((xyz, labels))

    def filter_points_by_label(self, label):
        """
        Filter the point cloud to only include points with the specified label.
        :param label: The label to filter by.
        :return: A NumPy array of points with the specified label.
        """
        return self.point_clouds[self.point_clouds[:, 3] == label]

    def find_median_z(self, points):
        """
        Find the median z-value from a subset of points.
        :param points: A NumPy array of points.
        :return: The median z-value.
        """
        return np.median(points[:, 2])

    def split_points_by_z(self, median_z):
        """
        Split the point cloud into top and bottom parts based on z-value.
        :param median_z: The median z-value used as the threshold.
        :return: Two NumPy arrays representing the top and bottom parts of the point cloud.
        """
        top_part = self.point_clouds[self.point_clouds[:, 2] >= median_z]
        bottom_part = self.point_clouds[self.point_clouds[:, 2] < median_z]
        return top_part, bottom_part

    def process(self, label=3):
        """
        Process the point cloud to filter by a specific label, find the median z-value, and split the cloud.
        :param label: Label to filter by (default is 3).
        :return: Two NumPy arrays representing the top and bottom parts of the point cloud.
        """
        # Filter points by the specified label
        label_points = self.filter_points_by_label(label)

        # Find the median z-value of these points
        median_z = self.find_median_z(label_points)

        # Split the entire point cloud based on this median z-value
        top_part, bottom_part = self.split_points_by_z(median_z)

        return top_part, bottom_part

    def visualize(self, top_part, bottom_part):
        """
        Visualize the top and bottom parts of the point cloud using different colors.
        """
        # Create a scatter plot for the top part
        trace1 = go.Scatter3d(
            x=top_part[:, 0], y=top_part[:, 1], z=top_part[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',  # Blue for top part
                opacity=0.8
            ),
            name='Top Part'
        )

        # Create a scatter plot for the bottom part
        trace2 = go.Scatter3d(
            x=bottom_part[:, 0], y=bottom_part[:, 1], z=bottom_part[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='red',  # Red for bottom part
                opacity=0.8
            ),
            name='Bottom Part'
        )

        # Layout configuration
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title='X Axis'),
                yaxis=dict(title='Y Axis'),
                zaxis=dict(title='Z Axis')
            )
        )

        # Combine traces in a figure
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig.show()

# Example usage
processor = PointCloudProcessor(num_points=1000)  # Change it: I created a processor with 1000 random points
top_part, bottom_part = processor.process(label=3)  # Process the cloud for label 3

# Visualize the results
processor.visualize(top_part, bottom_part)

