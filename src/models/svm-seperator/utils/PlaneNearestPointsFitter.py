import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import re


class PlaneNearestPointsFitter:
    def __init__(self, plane_equation, point_cloud):
        """
        Initializes the fitter with a plane equation and point cloud.

        :param plane_equation: A string formatted as "{}x + {}y + {}z + {} = 0"
                               representing the plane equation.
        :param point_cloud: A numpy array of shape (n_points, 3) representing the point cloud.
        """
        # Use regular expressions to extract the coefficients
        self.plane_coefficients = np.array(
            [float(re.search(r'([-+]?\d*\.?\d+)(?=[xyz])', term).group()) for term in plane_equation.split(' + ') if
             'x' in term or 'y' in term or 'z' in term])
        self.constant_term = float(re.search(r'[-+]?\d*\.?\d+$', plane_equation.split(' + ')[-1]).group())
        self.point_cloud = point_cloud

    def find_k_nearest_points(self, k):
        """
        Finds the k-nearest points to the plane.

        :param k: The number of nearest points to find
        :return: A numpy array of the k-nearest points
        """
        # Unpack the plane coefficients
        a, b, c = self.plane_coefficients
        d = self.constant_term

        # Calculate the distance of each point from the plane
        distances = np.abs(
            a * self.point_cloud[:, 0] + b * self.point_cloud[:, 1] + c * self.point_cloud[:, 2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)

        # Get indices of the k nearest points
        nearest_indices = np.argsort(distances)[:k]
        return self.point_cloud[nearest_indices]

    def fit_line_to_points(self, points):
        """
        Fits a line to the provided points using PCA to find the best-fit line in 3D space.

        :param points: A numpy array of shape (k, 3) representing the points to fit the line to.
        :return: A tuple containing the direction vector and a point on the line.
        """
        pca = PCA(n_components=1)
        pca.fit(points)

        # The direction vector of the line (principal component)
        direction_vector = pca.components_[0]

        # A point on the line (mean of the points)
        point_on_line = np.mean(points, axis=0)

        return direction_vector, point_on_line

    def process(self, k):
        """
        Processes the point cloud by finding the k-nearest points to the plane and fitting a line to them.

        :param k: The number of nearest points to consider.
        :return: The k-nearest points, direction vector of the fitted line, and a point on the line.
        """
        k_nearest_points = self.find_k_nearest_points(k)
        direction_vector, point_on_line = self.fit_line_to_points(k_nearest_points)
        return k_nearest_points, direction_vector, point_on_line

    def visualize(self, k_nearest_points, direction_vector, point_on_line):
        """
        Creates an interactive 3D plot of the point cloud, the nearest points, the fitted line, and the plane.

        :param k_nearest_points: The k-nearest points used for fitting the line.
        :param direction_vector: The direction vector of the fitted line.
        :param point_on_line: A point on the fitted line.
        """
        # Create a 3D scatter plot of the point cloud
        fig = go.Figure(data=[go.Scatter3d(x=self.point_cloud[:, 0],
                                           y=self.point_cloud[:, 1],
                                           z=self.point_cloud[:, 2],
                                           mode='markers',
                                           marker=dict(size=4, color='blue'),
                                           name='Point Cloud')])

        # Add the k-nearest points
        fig.add_trace(go.Scatter3d(x=k_nearest_points[:, 0],
                                   y=k_nearest_points[:, 1],
                                   z=k_nearest_points[:, 2],
                                   mode='markers',
                                   marker=dict(size=6, color='red'),
                                   name=f'{k_nearest_points.shape[0]} Nearest Points'))

        # Generate points for the fitted line
        t = np.linspace(-0.5, 0.5, 100)
        line_points = point_on_line + t[:, np.newaxis] * direction_vector

        fig.add_trace(go.Scatter3d(x=line_points[:, 0],
                                   y=line_points[:, 1],
                                   z=line_points[:, 2],
                                   mode='lines',
                                   line=dict(color='green', width=4),
                                   name='Fitted Line'))

        # Add the plane
        a, b, c = self.plane_coefficients
        d = self.constant_term

        # Generate a grid of x and y values
        x_plane = np.linspace(min(self.point_cloud[:, 0]), max(self.point_cloud[:, 0]), 10)
        y_plane = np.linspace(min(self.point_cloud[:, 1]), max(self.point_cloud[:, 1]), 10)
        x_plane, y_plane = np.meshgrid(x_plane, y_plane)
        z_plane = (-a * x_plane - b * y_plane - d) / c  # Plane equation solved for z

        fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_plane, colorscale='Viridis', opacity=0.5, name='Plane'))

        # Set plot labels and title
        fig.update_layout(scene=dict(xaxis_title='X',
                                     yaxis_title='Y',
                                     zaxis_title='Z'),
                          title="Interactive 3D Visualization of Point Cloud, Nearest Points, Fitted Line, and Plane")

        fig.show()


# Example usage
if __name__ == "__main__":
    # Define a plane equation in the specified format, e.g., "1x + 2y + -3z + 4 = 0"
    plane_eq = "1x + 2y + -3z + 4 = 0"

    # Create a random point cloud
    point_cloud = np.random.rand(100, 3)  # 100 points in 3D space

    # Initialize the fitter
    fitter = PlaneNearestPointsFitter(plane_eq, point_cloud)

    # Find the k-nearest points and fit a line
    k = 5
    k_nearest_points, direction_vector, point_on_line = fitter.process(k)

    # Print the result
    print("Direction vector of the fitted line:", direction_vector)
    print("A point on the fitted line:", point_on_line)

    # Visualize the result
    fitter.visualize(k_nearest_points, direction_vector, point_on_line)
