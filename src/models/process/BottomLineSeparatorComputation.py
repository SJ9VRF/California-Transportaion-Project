import numpy as np
import utils
import visualise_no_normalization as visualise
from PointCloudClassifier import PointCloudClassifier
from HyperplaneOperations import HyperplaneOperations
from ArrayPointCloudClassifier import ArrayPointCloudClassifier
from LabeledPointCloudProcessor import LabeledPointCloudProcessor
from PlaneNearestPointsFitter import PlaneNearestPointsFitter

# from PointCloudProcessor import PointCloudProcessor
# from SimplePointCloudProcessor import SimplePointCloudProcessor

# Step 1: Compute Center Part

# 1.1 Compute Left Plane Separator
file_path = "path_to_pc_file"  # Replace with actual path
file_path = visualise.file_paths[0]  # "path_to_pc_file"  # Replace with actual path
classifier = PointCloudClassifier(file_path)

classification_value_center = 3
classification_value_left = 1

X_center = classifier.load_point_cloud_data(classification_value_center)
X_left = classifier.load_point_cloud_data(classification_value_left)

y_center = np.ones(X_center.shape[0])
y_left = np.zeros(X_left.shape[0])


X = np.vstack((X_center, X_left))
y = np.hstack((y_center, y_left))

w_left, b_left = classifier.train_classifier(X, y)
eq_left = f"{w_left[0]}x + {w_left[1]}y + {w_left[2]}z + {b_left} = 0"

classifier.plot_decision_boundary(X, X_center, X_left, w_left, b_left)

print('----- Step 1.1 Completed! -----')

# 1.2 Compute Right Plane Separator
classification_value_right = 2

X_right = classifier.load_point_cloud_data(classification_value_right)
#X_center = classifier.load_point_cloud_data(classification_value_center)

y_right = np.zeros(X_right.shape[0])
#y_center = np.ones(X_center.shape[0])

X = np.vstack((X_center, X_right))
y = np.hstack((y_center, y_right))

w_right, b_right = classifier.train_classifier(X, y)
eq_right = f"{w_right[0]}x + {w_right[1]}y + {w_right[2]}z + {b_right} = 0"
classifier.plot_decision_boundary(X, X_center, X_right, w_right, b_right)

print('----- Step 1.2 Completed! -----')

# 1.3 Compute Points Between Hyperplanes

classification_value_around = 0
X_around = classifier.load_point_cloud_data(classification_value_around)
all_points = np.vstack((X_around, X_center, X_left, X_right))
all_labels = np.hstack((np.full(len(X_around), 0), np.full(len(X_center), 3), np.full(len(X_left), 2), np.full(len(X_right), 3)))
#all_points = np.vstack((X_around, X_center))
h_operations = HyperplaneOperations()
filtered_points, filtered_points_labels = h_operations.points_between_hyperplanes(all_points, all_labels, eq_left, eq_right)

h_operations.visualize(all_points, eq_left, eq_right, filtered_points, filtered_points_labels)
print('----- Step 1.3 Completed! -----')

# Step 2: Compute Bottom Part
#result = np.array([pt for pt in all_points if not any(np.array_equal(pt, pt2) for pt2 in filtered_points)])

# Result Processing
# Convert each array to a structured array with a compound data type
result, result_labels = utils.get_filtered_points_and_labels(all_points, filtered_points, all_labels)


# Main Step 2

processor = LabeledPointCloudProcessor(point_cloud=result, point_cloud_label=result_labels)  # Example with 1000 random points
(top_part, top_labels), (bottom_part, bottom_labels) = processor.process()
processor.visualize((top_part, top_labels), (bottom_part, bottom_labels))

print('----- Step 2 Completed! -----')

# Todo: gray and center classifir of the the bottom
# for  the  bottom  part, check the labels labels (0, 3) and partition into two groups and then call array classifier
# return the plane separator
# return fitted line

# Step 3: Compute SVM Separator for Bottom Part
"""
classification_value_bottom = 0

X_bottom = classifier.load_point_cloud_data(classification_value_bottom)
X_center = bottom_part

y_bottom = np.zeros(X_bottom.shape[0])
y_center = np.ones(X_center.shape[0])

X = np.vstack((X_bottom, X_center))
y = np.hstack((y_bottom, y_center))

w_bottom, b_bottom = classifier.train_classifier(X, y)
eq_bottom = f"{w_bottom[0]}x + {w_bottom[1]}y + {w_bottom[2]}z + {b_bottom} = 0"
"""

###

classifier = ArrayPointCloudClassifier()

array_label_0 = bottom_part[bottom_labels == 0]  # Create a new array for label 0 (Around)
array_label_3 = bottom_part[bottom_labels == 3]  # Create a new array for label 3 (Center)

y_a = np.zeros(array_label_0.shape[0])
y_b = 3 * np.ones(array_label_3.shape[0])

X = np.vstack((array_label_0, array_label_3))
y = np.hstack((y_a, y_b))

w_bottom, b_bottom = classifier.train_classifier(X, y)
eq_bottom = f"{w_bottom[0]}x + {w_bottom[1]}y + {w_bottom[2]}z + {b_bottom} = 0"
classifier.plot_decision_boundary(X, array_label_0, array_label_3, w_bottom, b_bottom)
utils.format_plane_equation(w_bottom, b_bottom)

print('----- Step 3 Completed! -----')

# Step 4: Compute Intersection - Method 1
fitter = PlaneNearestPointsFitter(eq_bottom, bottom_part)
k_nearest_points, direction_vector, point_on_line, t_range = fitter.process(k=5)

# Visualize the result
fitter.visualize(k_nearest_points, direction_vector, point_on_line, t_range)
print("Direction vector of the fitted line:", direction_vector)
print("A point on the fitted line:", point_on_line)


# Output Results
print()
print('Output Results')
print("Left Plane Equation:", eq_left)
print("Right Plane Equation:", eq_right)
print("Bottom Plane Equation:", eq_bottom)
print("Direction vector of the fitted line:", direction_vector)
print("A point on the fitted line:", point_on_line)
"""
# Optional: Visualization

classifier = PointCloudClassifier(file_path)
classifier.plot_decision_boundary(X_left, X_center, w_left, b_left)
classifier.plot_decision_boundary(X_right, X_center, w_right, b_right)
h_operations.visualize(filtered_points, eq_left, eq_right, filtered_points)
processor.visualize(top_part=None, bottom_part=bottom_part)
"""


print('----- Step 4 Completed! -----')
