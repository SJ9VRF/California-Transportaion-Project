import numpy as np
import plotly.graph_objects as go


class LabeledPointCloudProcessor:
    def __init__(self, point_cloud=None, point_cloud_label=None, num_points=1000):
        """
        Initialize the LabeledPointCloudProcessor with an existing point cloud and labels or generate a random one.
        :param point_cloud: An existing NumPy array representing the point cloud.
        :param point_cloud_label: An existing NumPy array representing the labels for the point cloud.
        :param num_points: Number of points to generate if no point cloud is provided.
        """
        if point_cloud is not None and point_cloud_label is not None:
            self.point_clouds = np.hstack((point_cloud, point_cloud_label.reshape(-1, 1)))
        else:
            self.point_clouds = self.generate_random_point_cloud(num_points)

    def generate_random_point_cloud(self, num_points):
        """
        Generate a random point cloud with x, y, z coordinates and labels.
        :param num_points: Number of points to generate.
        :return: A NumPy array representing the point cloud with labels.
        """
        xyz = np.random.rand(num_points, 3) * 100  # Random x, y, z values
        labels = np.random.randint(0, 2, num_points)  # Random labels (0 or 1)
        return np.hstack((xyz, labels.reshape(-1, 1)))

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
        :return: Two tuples of NumPy arrays representing the top and bottom parts of the point cloud and their labels.
        """
        top_part = self.point_clouds[self.point_clouds[:, 2] >= median_z]
        bottom_part = self.point_clouds[self.point_clouds[:, 2] < median_z]
        return top_part[:, :3], top_part[:, 3], bottom_part[:, :3], bottom_part[:, 3]

    def process(self):
        """
        Process the point cloud to find the median z-value and split the cloud.
        :return: Two tuples of NumPy arrays representing the top and bottom parts of the point cloud and their labels.
        """
        # Find the median z-value of all points
        median_z = self.find_median_z(self.point_clouds)

        # Split the entire point cloud based on this median z-value
        top_points, top_labels, bottom_points, bottom_labels = self.split_points_by_z(median_z)

        return (top_points, top_labels), (bottom_points, bottom_labels)

    def visualize(self, top_part, bottom_part):
        """
        Visualize the top and bottom parts of the point cloud using different colors.
        :param top_part: Tuple of the top points and their labels.
        :param bottom_part: Tuple of the bottom points and their labels.
        """
        top_points, top_labels = top_part
        bottom_points, bottom_labels = bottom_part

        # Create a scatter plot for the top part
        trace1 = go.Scatter3d(
            x=top_points[:, 0], y=top_points[:, 1], z=top_points[:, 2],
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
            x=bottom_points[:, 0], y=bottom_points[:, 1], z=bottom_points[:, 2],
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
if __name__ == "__main__":
    # Example point cloud data and labels: Replace these with your actual data
    result = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    result_labels = np.array([0, 1, 0])

    # Initialize the processor with the point cloud and labels
    processor = LabeledPointCloudProcessor(point_cloud=result, point_cloud_label=result_labels)

    # Process the cloud
    (top_part, top_labels), (bottom_part, bottom_labels) = processor.process()

    # Visualize the results
    processor.visualize((top_part, top_labels), (bottom_part, bottom_labels))

