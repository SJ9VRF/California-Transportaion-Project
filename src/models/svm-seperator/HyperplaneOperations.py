import numpy as np
import re
import plotly.graph_objects as go

"""
Short Comment:
# Given two hyperplane equations as input, return the points of intersection (if any) between them.
"""


class HyperplaneOperations:
    def __init__(self):
        pass

    @staticmethod
    def parse_hyperplane(equation):
        # Extracts the numerical values for coefficients and intercepts using regex
        coeffs = re.findall(r"([-+]?\d*\.?\d+)", equation)
        coeffs = [float(c) for c in coeffs]
        return np.array(coeffs[:3]), coeffs[3]

    @staticmethod
    def points_between_hyperplanes(points, eq1, eq2):
        w1, b1 = HyperplaneOperations.parse_hyperplane(eq1)
        w2, b2 = HyperplaneOperations.parse_hyperplane(eq2)
        # Computes dot product and adds intercept to determine side of plane
        hyperplane1_values = points.dot(w1) + b1
        hyperplane2_values = points.dot(w2) + b2
        # Masks for points between hyperplanes
        mask = ((hyperplane1_values > 0) & (hyperplane2_values < 0)) | ((hyperplane1_values < 0) & (hyperplane2_values > 0))
        return points[mask]

    @staticmethod
    def generate_sample_points(num_points=500):
        # Generates random points in a specified range
        return np.random.uniform(-10, 10, (num_points, 3))

    @staticmethod
    def visualize(points, eq1, eq2, filtered_points):
        w1, b1 = HyperplaneOperations.parse_hyperplane(eq1)
        w2, b2 = HyperplaneOperations.parse_hyperplane(eq2)

        # Setup the plot grid
        xx, yy = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
        zz1 = (-w1[0] * xx - w1[1] * yy - b1) / (w1[2] if w1[2] != 0 else 1)
        zz2 = (-w2[0] * xx - w2[1] * yy - b2) / (w2[2] if w2[2] != 0 else 1)

        fig = go.Figure()

        # All points
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                   mode='markers', marker=dict(size=2, color='blue'), name='All Points'))
        # Filtered points
        fig.add_trace(go.Scatter3d(x=filtered_points[:, 0], y=filtered_points[:, 1], z=filtered_points[:, 2],
                                   mode='markers', marker=dict(size=3, color='red'), name='Filtered Points'))
        # Planes
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz1, colorscale=[[0, 'green'], [1, 'green']], opacity=0.5, name='Hyperplane 1'))
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz2, colorscale=[[0, 'purple'], [1, 'purple']], opacity=0.5, name='Hyperplane 2'))

        # Setting layout for clear visibility and interactivity
        fig.update_layout(title='Points and Hyperplanes Visualization',
                          scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                          margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

# My Example usage
h_operations = HyperplaneOperations()
sample_points = h_operations.generate_sample_points()
eq1 = "1x + 0y + 2z - 1 = 0"
eq2 = "2x + 2y + 3z - 2 = 0"
filtered_points = h_operations.points_between_hyperplanes(sample_points, eq1, eq2)
h_operations.visualize(sample_points, eq1, eq2, filtered_points)

# More detailed instruction
#The red points are the points located between two planes (in the visualization).
