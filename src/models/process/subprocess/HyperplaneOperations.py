import numpy as np
import re
import plotly.graph_objects as go

class HyperplaneOperations:
    def __init__(self):
        pass

    @staticmethod
    def parse_hyperplane(equation):
        coeffs = re.findall(r"([-+]?\d*\.?\d+)", equation)
        coeffs = [float(c) for c in coeffs]
        return np.array(coeffs[:3]), coeffs[3]

    @staticmethod
    def points_between_hyperplanes(points, labels, eq1, eq2):
        w1, b1 = HyperplaneOperations.parse_hyperplane(eq1)
        w2, b2 = HyperplaneOperations.parse_hyperplane(eq2)
        hyperplane1_values = points.dot(w1) + b1
        hyperplane2_values = points.dot(w2) + b2
        mask = ((hyperplane1_values > 0) & (hyperplane2_values < 0)) | \
               ((hyperplane1_values < 0) & (hyperplane2_values > 0))
        return points[mask], labels[mask]

    @staticmethod
    def generate_sample_points(num_points=500):
        points = np.random.uniform(-10, 10, (num_points, 3))
        labels = np.random.randint(0, 100, num_points)  # Assuming labels are integers from 0 to 99
        return points, labels

    @staticmethod
    def visualize(points, eq1, eq2, filtered_points, filtered_labels):
        w1, b1 = HyperplaneOperations.parse_hyperplane(eq1)
        w2, b2 = HyperplaneOperations.parse_hyperplane(eq2)

        # Determine the range of the mesh grid based on the points
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        # Create a mesh grid based on the range of points
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        zz1 = (-w1[0] * xx - w1[1] * yy - b1) / (w1[2] if np.abs(w1[2]) > 1e-6 else 1e-6)
        zz2 = (-w2[0] * xx - w2[1] * yy - b2) / (w2[2] if np.abs(w2[2]) > 1e-6 else 1e-6)

        fig = go.Figure()

        # Plot all points
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                   mode='markers', marker=dict(size=2, color='blue'), name='All Points'))

        # Plot filtered points with labels as hover text
        fig.add_trace(go.Scatter3d(x=filtered_points[:, 0], y=filtered_points[:, 1], z=filtered_points[:, 2],
                                   mode='markers', marker=dict(size=3, color='red'),
                                   text=filtered_labels, hoverinfo='text', name='Filtered Points'))

        # Plot the hyperplanes
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz1, colorscale=[[0, 'green'], [1, 'green']], opacity=0.5, name='Hyperplane 1'))
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz2, colorscale=[[0, 'purple'], [1, 'purple']], opacity=0.5, name='Hyperplane 2'))

        # Update layout for clarity and better visualization
        fig.update_layout(title='3D Visualization of Points and Hyperplanes',
                          scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                          margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

# Example usage
if __name__ == "__main__":
    h_operations = HyperplaneOperations()
    sample_points, sample_labels = h_operations.generate_sample_points()
    eq1 = "1x + 0y + 2z - 1 = 0"
    eq2 = "2x + 2y + 3z - 2 = 0"
    filtered_points, filtered_labels = h_operations.points_between_hyperplanes(sample_points, sample_labels, eq1, eq2)
    h_operations.visualize(sample_points, eq1, eq2, filtered_points, filtered_labels)

