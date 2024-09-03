import numpy as np

def get_filtered_points_and_labels(all_points, filtered_points, all_labels):
    # Create a structured dtype based on the dtype of all_points
    dtype = [('x', all_points.dtype), ('y', all_points.dtype), ('z', all_points.dtype)]

    # Use np.setdiff1d to remove points in filtered_points from all_points
    result_view = np.setdiff1d(all_points.view(dtype), filtered_points.view(dtype))

    # Convert the structured array back to a regular ndarray
    result = result_view.view(all_points.dtype).reshape(-1, 3)

    # Find the indices in all_points that correspond to the result array
    result_indices = np.nonzero(np.in1d(all_points.view(dtype), result_view))[0]

    # Use these indices to extract the corresponding labels from all_labels
    result_labels = all_labels[result_indices]

    # Return the result points and their corresponding labels
    return result, result_labels

# Example usage:
# all_points = np.array([...])  # Define your array of points
# filtered_points = np.array([...])  # Define the array of points to filter out
# all_labels = np.array([...])  # Define the corresponding labels

# result, result_labels = get_filtered_points_and_labels(all_points, filtered_points, all_labels)

def format_plane_equation(w, b):
    """
    Formats and prints the plane equation in the form of ax + by + cz + d = 0.
    :param w: Coefficient array [a, b, c] for x, y, z respectively.
    :param b: Bias term d.
    :return: A formatted string representing the plane equation.
    """
    terms = []

    # Add the x, y, z terms with appropriate signs
    if w[0] != 0:
        terms.append(f"{w[0]}x")
    if w[1] != 0:
        terms.append(f"{'+' if w[1] > 0 else ''}{w[1]}y")
    if w[2] != 0:
        terms.append(f"{'+' if w[2] > 0 else ''}{w[2]}z")

    # Add the bias term with appropriate sign
    if b != 0:
        terms.append(f"{'+' if b > 0 else ''}{b}")

    # Combine all terms into a single equation string
    equation = " ".join(terms) + " = 0"

    return equation

"""
# Example usage:
w = [2, -3, 5]
b = -4

equation = format_plane_equation(w, b)
print(equation)
"""
