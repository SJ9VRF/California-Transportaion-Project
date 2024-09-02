import numpy as np
import plotly.graph_objects as go
import re


class PlaneIntersectionFinder:
    def __init__(self, plane_equation_a, plane_equation_b):
        """
        Initializes the finder with two plane equations.

        :param plane_equation_a: A string formatted as "{}x + {}y + {}z + {} = 0"
                                 representing the first plane equation (plane separator).
        :param plane_equation_b: A string formatted as "{}x + {}y + {}z + {} = 0"
                                 representing the second plane equation (ramp plane).
        """
        # Extract coefficients from the plane equations
        self.plane_a_coefficients = self._extract_coefficients(plane_equation_a)
        self.plane_b_coefficients = self._extract_coefficients(plane_equation_b)

    def _extract_coefficients(self, plane_equation):
        """
        Extracts coefficients from a plane equation string.

        :param plane_equation: A string formatted as "{}x + {}y + {}z + {} = 0".
        :return: A tuple (a, b, c, d) where ax + by + cz + d = 0.
        """
        coefficients = np.array(
            [float(re.search(r'([-+]?\d*\.?\d+)(?=[xyz])', term).group()) for term in plane_equation.split(' + ') if
             'x' in term or 'y' in term or 'z' in term])
        constant_term = float(re.search(r'[-+]?\d*\.?\d+$', plane_equation.split(' + ')[-1]).group())
        return np.append(coefficients, constant_term)

    def find_intersection_line(self):
        """
        Finds the line of intersection between the two planes.

        :return: A string formatted as "{}x + {}y + {}z = {}" representing the line of intersection.
        """
        # Unpack coefficients for both planes
        a1, b1, c1, d1 = self.plane_a_coefficients
        a2, b2, c2, d2 = self.plane_b_coefficients

        # The direction vector of the intersection line is the cross product of the normals of the two planes
        direction_vector = np.cross([a1, b1, c1], [a2, b2, c2])

        # Find a point on the intersection line by solving the system of plane equations
        A = np.array([[a1, b1, c1],
                      [a2, b2, c2],
                      [direction_vector[0], direction_vector[1], direction_vector[2]]])
        B = np.array([-d1, -d2, 0])

        # Solve for the intersection point
        if np.linalg.matrix_rank(A) == 3:
            point_on_line = np.linalg.solve(A, B)
        else:
            # If the matrix is singular, the planes are parallel and do not intersect
            raise ValueError("The planes do not intersect or are parallel.")

        # Construct the equation of the line in the requested format
        line_equation = "{}x + {}y + {}z = {}".format(direction_vector[0], direction_vector[1], direction_vector[2],
                                                      np.dot(direction_vector, point_on_line))
        return line_equation, direction_vector, point_on_line

    def visualize(self, direction_vector, point_on_line):
        """
        Creates an interactive 3D plot of the two planes and their intersection line.

        :param direction_vector: The direction vector of the intersection line.
        :param point_on_line: A point on the intersection line.
        """
        # Generate points for the intersection line
        t = np.linspace(-0.5, 0.5, 100)
        line_points = point_on_line + t[:, np.newaxis] * direction_vector

        # Create a 3D scatter plot of the intersection line
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=line_points[:, 0],
                                   y=line_points[:, 1],
                                   z=line_points[:, 2],
                                   mode='lines',
                                   line=dict(color='green', width=4),
                                   name='Intersection Line'))

        # Add the first plane (plane separator)
        a1, b1, c1, d1 = self.plane_a_coefficients[:4]
        x_plane = np.linspace(-10, 10, 10)
        y_plane = np.linspace(-10, 10, 10)
        x_plane, y_plane = np.meshgrid(x_plane, y_plane)
        z_plane = (-a1 * x_plane - b1 * y_plane - d1) / c1

        fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_plane, colorscale='Viridis', opacity=0.5, name='Plane A'))

        # Add the second plane (ramp plane)
        a2, b2, c2, d2 = self.plane_b_coefficients[:4]
        z_plane = (-a2 * x_plane - b2 * y_plane - d2) / c2

        fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_plane, colorscale='Cividis', opacity=0.5, name='Plane B'))

        # Set plot labels and title
        fig.update_layout(scene=dict(xaxis_title='X',
                                     yaxis_title='Y',
                                     zaxis_title='Z'),
                          title="Intersection of Two Planes")

        fig.show()


# Example usage
if __name__ == "__main__":
    # Define two plane equations
    plane_eq_a = "1x + 2y + -3z + 4 = 0"  # Plane separator
    plane_eq_b = "2x + -1y + 1z + 3 = 0"  # Ramp plane

    # Initialize the finder
    finder = PlaneIntersectionFinder(plane_eq_a, plane_eq_b)

    # Find the line of intersection between the two planes
    line_equation, direction_vector, point_on_line = finder.find_intersection_line()

    # Print the result
    print("Intersection line equation:", line_equation)

    # Visualize the intersection
    finder.visualize(direction_vector, point_on_line)
