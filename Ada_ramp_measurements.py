"""
Debug Summary
1- check gutter classffication
2- Check threshold, it doesn't work even normalisation
3- New point-cloud file - units?

5- He has different classification, 5
center, left, right, gutter, other points

Failed algorithm ...
4- You are not sure about p1, p2, p3, p4 ... and right location spot
# messed up regaring the order of the points ....
"""


import laspy
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from scipy.spatial import distance
import math
import random

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import combinations
from tqdm import tqdm
from math import comb

random.seed(21)


activated_fit_line_with_pca_threshold = True
activated_fit_line_with_pca_threshold_z = False

## Find the nearest points to a line segment
def nearest_points(p1, p2, point_cloud, num_points=200, gutter=False):
    if gutter:
        num_points = 100    ## HARDCODED. NUM CANDIDATE POINTS FOR GUTTER
    line_direction = p2 - p1
    line_length = np.linalg.norm(line_direction)
    unit_line_direction = line_direction / line_length
    vectors_to_points = point_cloud - p1
    dot_products = np.dot(vectors_to_points, unit_line_direction)
    projected_points = p1 + np.clip(dot_products[:, np.newaxis], 0, line_length) * unit_line_direction
    distances = np.linalg.norm(projected_points - point_cloud, axis=1)
    nearest_point_indices = np.argsort(distances)[:num_points]

    return point_cloud[nearest_point_indices]

## Divide a line segment into n(10) segments
def divide_line_into_segments(p_start, p_end, num_segments=10):

    return np.array([p_start + (p_end - p_start) * t for t in np.linspace(0, 1, num_segments + 1)])

## Not used. Divide a line segment into n(100) segments
def divide_line_into_segments_100(p_start, p_end, num_segments=100):

    return np.array([p_start + (p_end - p_start) * t for t in np.linspace(0, 1, num_segments + 1)])

## Calculate slope of a line (two points)
def calculate_slope_3d(point_1, point_2):
    delta_x = point_2[0] - point_1[0]
    delta_y = point_2[1] - point_1[1]
    delta_z = point_2[2] - point_1[2]

    if delta_x == 0 and delta_y == 0:
        return float('inf')
    else:
        return round(delta_z / np.sqrt(delta_x**2 + delta_y**2)*100, 1)

## Calculate slope of pca model    
def calculate_slope_pca(pca_model):
    direction_vector = pca_model.components_[0]
    dx, dy, dz = direction_vector
    if dx == 0 and dy == 0:
        slope = float('inf')
    else:
        slope = dz / np.sqrt(dx**2 + dy**2) 

    slope_percentage = abs(round(slope * 100, 1))

    return slope_percentage

## Helper function (update masked points)
def update_mask(original_mask, new_mask):
    indices = np.where(original_mask)[0]
    updated_mask = np.copy(original_mask)
    updated_mask[indices] = updated_mask[indices] & new_mask
    return updated_mask


## Fit a line with PCA with threshold
def fit_line_with_pca_threshold(X, y, threshold=0.125, max_iterations=10):
#def fit_line_with_pca_threshold(X, y, threshold=10, max_iterations=10):
    threshold = 125 # Suggested by kin
    max_iterations = 10
    data = np.hstack((X, y[:, np.newaxis]))
    cul_mask = np.ones(len(data), dtype=bool)
    
    iteration = 0
    while iteration < max_iterations:
        pca = PCA(n_components=1)
        pca.fit(data)
        projected = pca.transform(data)
        reconstructed = pca.inverse_transform(projected)
        residuals_xyz = np.sqrt(np.sum((data - reconstructed)**2, axis=1))
        print('residuals_xyz', np.min(residuals_xyz), np.max(residuals_xyz), np.mean(residuals_xyz))
        mask = residuals_xyz <= threshold
        refined_data = data[mask]

        print(f"Iteration {iteration+1}: Original dataset size: {len(data)}, Refined dataset size: {len(refined_data)}")
        
        if len(refined_data) < len(data):
            data = refined_data
        elif len(refined_data) == len(data):
            break
        
        iteration += 1

        cul_mask = update_mask(cul_mask, mask)
    
    return pca, cul_mask

## Fit a line with PCA with threshold in z direction
def fit_line_with_pca_threshold_z(X, y, cul_mask, z_threshold=0.021, max_iterations=10):
    z_threshold = 0.021
    max_iterations = 10
    data = np.hstack((X, y[:, np.newaxis]))
    data = data[cul_mask]

    iteration = 0
    while iteration < max_iterations:
        pca = PCA(n_components=1)
        pca.fit(data)
        projected = pca.transform(data)
        reconstructed = pca.inverse_transform(projected)
        residuals_z = np.abs(data[:, 2] - reconstructed[:, 2])

        mask = residuals_z <= z_threshold
        refined_data = data[mask]

        # print(f"z Iteration {iteration+1}: Original dataset size: {len(data)}, Refined dataset size: {len(refined_data)}")
        
        if len(refined_data) < len(data):
            data = refined_data
        else:
            break
        
        iteration += 1

        cul_mask = update_mask(cul_mask, mask)
    
    return pca, cul_mask

## Project points to a line
def project_points_to_line(points, line_point, line_direction):
    line_direction_normalized = line_direction / np.linalg.norm(line_direction)
    projections = line_point + np.dot(points - line_point, line_direction_normalized[:, np.newaxis]) * line_direction_normalized

    return projections

## Find gaps in projections
def find_gaps_in_projections(projections, gap_threshold=0.25):
    holes = []
    distances = np.linalg.norm(np.diff(projections, axis=0), axis=1)
    hole_indices = np.where(distances > gap_threshold)[0]
    
    # print(sorted(distances)[::-1])
    
    for idx in hole_indices:
        holes.append((idx, idx + 1))

    return holes

def find_intersection(p1, d1, p2, d2):
    t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    intersection = p1 + d1 * t[0]
    
    return intersection

def find_z_for_xy_on_line(x, y, line_point, direction_vector):
    px, py, pz = line_point
    dx, dy, dz = direction_vector

    if dx != 0:
        t = (x - px) / dx
    elif dy != 0:
        t = (y - py) / dy
    else:
        raise ValueError("Line is vertical")

    z = pz + t * dz

    return z

def calculate_3d_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)*12

    return round(distance, 1)

def yes_no(tf):
    if tf:
        return 'Yes'
    else:
        return 'No'
    
def normalize_points(corner_points, point_cloud_mean):
    return corner_points - point_cloud_mean

def normalize_point_cloud(point_cloud, point_cloud_mean):
    return point_cloud - point_cloud_mean

def crop_point_cloud(corner_points, point_cloud, padding):
    p1, p2, p3, p4 = corner_points
    min_x = min(p1[0], p2[0], p3[0], p4[0])
    max_x = max(p1[0], p2[0], p3[0], p4[0])
    min_y = min(p1[1], p2[1], p3[1], p4[1])
    max_y = max(p1[1], p2[1], p3[1], p4[1])

    filtered_points = point_cloud[(point_cloud[:, 0] >= min_x - padding) & (point_cloud[:, 0] <= max_x + padding) & 
                                (point_cloud[:, 1] >= min_y - padding) & (point_cloud[:, 1] <= max_y + padding)]

    sorted_points = filtered_points[np.argsort(filtered_points[:, 2])]

    return sorted_points

def solve_for_t(p1, p2, A, B, C, D):
    t = (D - A*p1[0] - B*p1[1] - C*p1[2]) / (A*(p2[0]-p1[0]) + B*(p2[1]-p1[1]) + C*(p2[2]-p1[2]))
    return t

def get_perpendicular(p34_interest, p1, p2, p3, p4):
    normal_vector = p4 - p3
    A, B, C = normal_vector
    D = np.dot(normal_vector, p34_interest)

    t = solve_for_t(p1, p2, A, B, C, D)
    intersection_point = p1 + t * (p2 - p1)
    direction_vector = intersection_point - p34_interest
    return intersection_point, direction_vector


def adjusted_points(p34_5, plane, p2close):
    p1, p2, p3, p4 = plane
    p34_5_intersection_point, p34_5_direction_vector = get_perpendicular(p34_5, p1, p2, p3, p4)
    ## p2 is closer
    if p2close:
        new_intersection_point, new_direction_vector = get_perpendicular(p2, p4, p1, p34_5_intersection_point, p34_5)
        # segments_p1_p4 = divide_line_into_segments(new_intersection_point, p4)
        # segments_p1_p4 = np.vstack((segments_p1_p4[1], segments_p1_p4[5], segments_p1_p4[9]))
        # segments_p2_p3 = divide_line_into_segments(p2, p3)
        # segments_p2_p3 = np.vstack((segments_p2_p3[1], segments_p2_p3[5], segments_p2_p3[9]))
        # segments_p1_p2 = divide_line_into_segments(new_intersection_point, p2)
        # segments_p4_p3 = divide_line_into_segments(p4, p3)

        return [new_intersection_point, p2, p3, p4]

    else: ## p3 is closer
        new_intersection_point, new_direction_vector = get_perpendicular(p1, p3, p2, p34_5_intersection_point, p34_5)
        return [p1, new_intersection_point, p3, p4]
    

def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area

def extract_corners(file_path):
    las = laspy.read(file_path)

    #target_classifications = [122, 123] # left + right + center part of the ramp
    target_classifications = [1, 2, 3]
    indices = np.isin(las.classification, target_classifications)
    filtered_points = las.points[indices]

    x = filtered_points.x
    y = filtered_points.y
    z = filtered_points.z

    points_2d_raw = np.vstack((x, y)).T
    points_3d_raw = np.vstack((x, y, z)).T

    print(type(points_3d_raw[:, :2]))

    hull = ConvexHull(points_3d_raw[:, :2])
    points_3d = points_3d_raw[hull.vertices]

    n = len(points_3d)
    k = 6
    num_combinations = comb(n, k)
    combos = combinations(points_3d, 6)
    max_area = 0
    best_polygon = None
    for combo in tqdm(combos, total=num_combinations, desc="Analyzing Combos"):
        combo = np.array(combo)
        try:
            hull = ConvexHull(combo[:, :2])
            if len(hull.vertices) == 6:
                area = polygon_area(combo[hull.vertices, :2])
                if area > max_area:
                    max_area = area
                    best_polygon = combo
        except ConvexHull.QHullError:
            continue

    plt.plot(points_2d_raw[:, 0], points_2d_raw[:, 1], 'o', markersize=1, alpha=0.3, label='All Points')
    if best_polygon is not None:
        plt.fill(best_polygon[:, 0], best_polygon[:, 1], 'r', alpha=0.5, label='Largest Area Polygon')
        plt.plot(best_polygon[:, 0], best_polygon[:, 1], 'ro', label='Corner Points', markersize=5)
    plt.title(f"Largest Area Polygon with Six Points (Area = {max_area:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()

    if best_polygon is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d_raw[:, 0], points_3d_raw[:, 1], points_3d_raw[:, 2], color='blue', label='All Points', s=1)

        ax.scatter(best_polygon[:, 0], best_polygon[:, 1], best_polygon[:, 2], color='red', label='Largest Polygon Corners')
        ax.set_title(f"Largest Area Polygon with Six Points")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")
        ax.legend()

        # ax.view_init(azim=45)

        plt.show()
    else:
        print("No valid convex hull was found.")

    for point in best_polygon:
        print("{:.2f}, {:.2f}, {:.2f}".format(point[0], point[1], point[2]))

    points_sorted_by_z = best_polygon[np.argsort(best_polygon[:, 2])]

    smallest_two_by_z = points_sorted_by_z[:2]
    smallest_two_by_z_sorted_by_x = smallest_two_by_z[np.argsort(smallest_two_by_z[:, 0])]

    remaining_points = points_sorted_by_z[2:]
    remaining_points_sorted_by_x = remaining_points[np.argsort(remaining_points[:, 0])]

    corner_points = []

    print("Two points with the smallest Z values, sorted by X:")
    for point in smallest_two_by_z_sorted_by_x:
        corner_points.append(np.array([point[0], point[1], point[2]]))
        print("{:.2f}, {:.2f}, {:.2f}".format(point[0], point[1], point[2]))
    print("Remaining four points, sorted by X:")
    for point in remaining_points_sorted_by_x:
        corner_points.append(np.array([point[0], point[1], point[2]]))
        print("{:.2f}, {:.2f}, {:.2f}".format(point[0], point[1], point[2]))
        

    return corner_points[3], corner_points[4], corner_points[1], corner_points[0], corner_points[2], corner_points[5]

def extract_gutter(file_path):

    # read file
    las = laspy.read(file_path)
    print('unique classification values', set(las.classification))
    # filter class
    # target_classifications = [125]
    target_classifications = [0, 1, 2, 3] # glutter class
    indices = np.isin(las.classification, target_classifications)
    filtered_points = las.points[indices]

    # processing points
    x, y, z = filtered_points.x, filtered_points.y, filtered_points.z

    points_2d_raw = np.vstack((x, y)).T
    points_3d_raw = np.vstack((x, y, z)).T

    print(type(points_3d_raw[:, :2]), points_3d_raw[:, :2].size)

    # Compute convex hull
    hull = ConvexHull(points_3d_raw[:, :2])
    points_3d = points_3d_raw[hull.vertices]

    # Compute convex hull (cont.)
    n = len(points_3d) # point counts
    k = 4  # try to find the 4 corner points of the convex hull
    num_combinations = comb(n, k)
    combos = combinations(points_3d, 4) # compute comb
    max_area = 0
    best_polygon = None
    for combo in tqdm(combos, total=num_combinations, desc="Analyzing Combos"):
        # compute all comb area and return the max area of convex hull based on the combs
        combo = np.array(combo)
        try:
            hull = ConvexHull(combo[:, :2])
            if len(hull.vertices) == 4:
                area = polygon_area(combo[hull.vertices, :2])
                if area > max_area:
                    max_area = area
                    best_polygon = combo
        except ConvexHull.QHullError:
            continue


    # Visualising the cornerpoints basd on convex hull
    plt.plot(points_2d_raw[:, 0], points_2d_raw[:, 1], 'o', markersize=1, alpha=0.3, label='All Points')
    if best_polygon is not None:
        plt.fill(best_polygon[:, 0], best_polygon[:, 1], 'r', alpha=0.5, label='Largest Area Polygon')
        plt.plot(best_polygon[:, 0], best_polygon[:, 1], 'ro', label='Corner Points', markersize=5)
    plt.title(f"Largest Area Polygon with Four Points (Area = {max_area:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()


    # issue 1: check hole
    # visualization of best convex hull
    if best_polygon is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d_raw[:, 0], points_3d_raw[:, 1], points_3d_raw[:, 2], color='blue', label='All Points', s=1)

        ax.scatter(best_polygon[:, 0], best_polygon[:, 1], best_polygon[:, 2], color='red', label='Largest Polygon Corners')
        ax.set_title(f"Largest Area Polygon with Six Points")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")
        ax.legend()

        # ax.view_init(azim=45)
        plt.show()
    else:
        print("No valid convex hull was found.")


    for point in best_polygon:  # print 4 corner points
        print("{:.2f}, {:.2f}, {:.2f}".format(point[0], point[1], point[2]))

    # sort by z
    points_sorted_by_z = best_polygon[np.argsort(best_polygon[:, 2])]

    # smallest by z
    smallest_two_by_z = points_sorted_by_z[:2]
    # then by x
    smallest_two_by_z_sorted_by_x = smallest_two_by_z[np.argsort(smallest_two_by_z[:, 0])]

    remaining_points = points_sorted_by_z[2:]  #  next points after two smallest
    # sort rem points by x
    remaining_points_sorted_by_x = remaining_points[np.argsort(remaining_points[:, 0])]

    gutter_points = [] # wanted to have sorted gutter points
    print("Two points with the smallest Z values, sorted by X:")
    for point in smallest_two_by_z_sorted_by_x:
        gutter_points.append(np.array([point[0], point[1], point[2]]))
        print("{:.2f}, {:.2f}, {:.2f}".format(point[0], point[1], point[2]))
    print("Remaining four points, sorted by X:")
    for point in remaining_points_sorted_by_x:
        gutter_points.append(np.array([point[0], point[1], point[2]]))
        print("{:.2f}, {:.2f}, {:.2f}".format(point[0], point[1], point[2]))

    return gutter_points[2], gutter_points[3], gutter_points[1], gutter_points[0]

if __name__ == '__main__':
    # las_file_path = 'extracted/HUM101_80_run1_Left_gnd_1_Output_0.las'
    las_file_path = 'extracted/HUM101_80_run1_Left_gnd_2_Output_0.las'  # tested code

    file_paths = ["ChicoADA144_r3_Left_GND_2.las",
                  "NAP121_6_Left_gnd_0.las",
                  'SF1_19_run3_1_Right_gnd_2_1.las',
                  'SUT20_20231019_run1_Right_1000m_GND_1.las',
                  'woodlandADA_r8_Left_GND_20.las',
                  'woodlandADA_r10_Left_GND_13.las']
    main_path = "/Users/arefehyavary/Downloads/pointclouds w updated classification/"
    las_file_path = main_path + file_paths[0]


    corner_points = extract_corners(las_file_path)
    gutter_points = extract_gutter(las_file_path)
    """
    corner_points = np.array([
         [62680.00, 630260.00, 15359.99],
         [65730.00, 632580.01, 15329.99],
         [69510.00, 628559.99, 14970.00],
         [66569.99, 625900.02, 14920.00],
         [62450.00, 625349.97, 15310.00],
         [70739.99, 631979.98, 15369.99],
         ])

    # file zero
    corner_points = np.array([
        [4298.014, 12800.14, 18.001],
        [9159.92, 21790.04, 840],
        [12399.1, 17700.31, 72],
        [11250, 7950.07, 2680.01],
        [4439.98, 2250, 47.001],
        [54490, 15880, 300.01],
    ])
    """

    # it is partitioning the edges
    selected_segments_ramp = [9, 5, 1]
    selected_segments_flare = [9, 5]
    selected_segments_gutter = [8, 2]

    
    in_file = laspy.read(las_file_path)
    point_cloud = np.vstack((in_file.x, in_file.y, in_file.z)).T
    point_cloud_mean = point_cloud.mean(axis=0)
    print(len(point_cloud))
    corner_points = normalize_points(corner_points, point_cloud_mean)
    point_cloud = normalize_point_cloud(point_cloud, point_cloud_mean)

    # using padding arround classfied ramp parts, make point cloud smaller
    point_cloud = crop_point_cloud(corner_points[:4], point_cloud, padding=30)

    print(len(point_cloud))

    p1, p2, p3, p4, fl, fr = corner_points
    g1, g2, g3, g4 = gutter_points


    # computing slopes
    p34 = p3-p4
    g34 = g3-g4

    p34_5 = (p3 + p4) / 2
    p34_1 = (1 * p3 + 9 * p4) / 10
    p34_9 = (9 * p3 + 1 * p4) / 10

    g34_5 = (g3 + g4) / 2
    g34_1 = (1 * g3 + 9 * g4) / 10
    g34_9 = (9 * g3 + 1 * g4) / 10

    ramp_points = [p1, p2, p3, p4]

    p1_proj = project_points_to_line(p1, p34_5, p34)
    p2_proj = project_points_to_line(p2, p34_5, p34)
    p2close = calculate_3d_distance(p1, p1_proj) > calculate_3d_distance(p2, p2_proj)
    if p2close:
        print('p2 is closer')
    else:
        print('p1 is closer')
    adj_ramp_points = adjusted_points(p34_5, ramp_points, p2close)
    p1, p2, p3, p4 = adj_ramp_points

    gutter_points = [g1, g2, g3, g4]

    g1_proj = project_points_to_line(g1, g34_5, g34)
    g2_proj = project_points_to_line(p2, g34_5, g34)

    # trying to see which top two points are far away from the bottom edge, to find g1 and g2
    g2close = calculate_3d_distance(g1, g1_proj) > calculate_3d_distance(g2, g2_proj)
    if g2close:
        print('g2 is closer')
    else:
        print('g1 is closer')

    # refining the corner edge to make that gutter and main ramp edges to paralell
    # for our purpose of the algorithm, ramps edges should be parallel to be able to compute this slope, in this algorithm
    # Type A ramp
    # Todo: check it later if needed
    adj_gutter_points = adjusted_points(g34_5, gutter_points, g2close)
    g1, g2, g3, g4 = adj_gutter_points
    # g1 and g4 is the left part, g2 and g3 is right part. g1 and g2 are top. g3 and g4 are bottom
    g1 = divide_line_into_segments_100(g1, g4)[25]  # extracting the right part, start_point, end_point, num_segments = 25,
    g2 = divide_line_into_segments_100(g2, g3)[25]  # extracting the left part


    """Segment all parts into 10 segments"""
    # p1 is the top left of the main ramp
    # p2 is the top right of the main ramp
    # p3 is the bottom right of the main ramp
    # p4 is the bottom left of the main ramp
    segments_p1_p4 = divide_line_into_segments(p1, p4)  # num_segments = 10, default value
    segments_p2_p3 = divide_line_into_segments(p2, p3)
    segments_p1_p2 = divide_line_into_segments(p1, p2)
    segments_p4_p3 = divide_line_into_segments(p4, p3)

    # fl the end point in the Left flare
    # fr the end point in the Right flare
    segments_p1_fl = divide_line_into_segments(p1, fl)
    segments_p2_fr = divide_line_into_segments(p2, fr)

    segments_g1_g4 = divide_line_into_segments(g1, g4)
    segments_g2_g3 = divide_line_into_segments(g2, g3)
    segments_g1_g2 = divide_line_into_segments(g1, g2)
    segments_g4_g3 = divide_line_into_segments(g4, g3)


    # _a:  A-slope calculation is the slope of the main ramp.
    nearest_points_a = np.empty((0, 3))
    nearest_points_a_masked = np.empty((0, 3))
    # Check if there is a hole in the slope line (it is not straight), we have to consider that curve in the
    # width calculation
    nearest_points_selected_segments_b_hole = np.empty((0, 3))
    nearest_points_selected_segments_b_masked_hole = np.empty((0, 3)) # mask purpose?

    # _b:  Cross slope: B-slope calculation is the slope of the main ramp.
    nearest_points_b = np.empty((0, 3))
    nearest_points_b_masked = np.empty((0, 3))

    # D: left side of the ramp
    nearest_points_d = np.empty((0, 3))
    nearest_points_d_masked = np.empty((0, 3))

    # E: right side of the ramp
    nearest_points_e = np.empty((0, 3))
    nearest_points_e_masked = np.empty((0, 3))

    # F: Gutter main slope
    nearest_points_f = np.empty((0, 3))
    nearest_points_f_masked = np.empty((0, 3))

    # G: Gutter cross slope
    nearest_points_g = np.empty((0, 3))
    nearest_points_g_masked = np.empty((0, 3))

    a_calc, b_calc, c_calc, d_calc, e_calc, f_calc, g_calc = [[] for _ in range(7)]
    #  line points for each part for computing width of each section
    aline_points, bline_points, dline_points, eline_points, fline_points, gline_points = [[] for _ in range(6)]
    #  for visualization purpose
    trace_a_lines, trace_b_lines, trace_d_lines, trace_e_lines, trace_f_lines, trace_g_lines = [[] for _ in range(6)]

    # USED FOR TESTING - Generate a fake hole on ramp
    # for k, i in enumerate([5]):
    #     segment_nearest_points_b_hole = nearest_points(segments_p1_p4[i], segments_p2_p3[i], point_cloud, 1000)
    #     nearest_points_selected_segments_b_hole = np.vstack((nearest_points_selected_segments_b_hole, segment_nearest_points_b_hole))


    #     X = segment_nearest_points_b_hole[:, :2]
    #     y = segment_nearest_points_b_hole[:, 2]
    #     pca, mask_thres = fit_line_with_pca_threshold(X, y)

    #     nearest_points_selected_segments_b_masked_hole = np.vstack((nearest_points_selected_segments_b_masked_hole, segment_nearest_points_b_hole[mask_thres]))
        
    #     mean = pca.mean_
    #     principal_direction = pca.components_[0]

    #     principal_direction_xy = principal_direction[:2]

    #     # Directions of bounding lines in XY plane
    #     d_p4_p1 = p1[:2] - p4[:2]
    #     d_p2_p3 = p3[:2] - p2[:2]

    #     # Finding intersections in the XY plane
    #     intersection1 = find_intersection(p4[:2], d_p4_p1, mean[:2], principal_direction_xy)
    #     intersection2 = find_intersection(p2[:2], d_p2_p3, mean[:2], principal_direction_xy)

    #     line_point = np.array(mean)  # PCA mean as line_point
    #     direction_vector = np.array(principal_direction)  # PCA first component as direction_vector
    #     x1, y1 = intersection1
    #     x2, y2 = intersection2

    #     z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)

    #     intersection_point_1 = np.array([x1, y1, z1])
    #     distances = np.array([distance.euclidean(intersection_point_1, point) for point in segment_nearest_points_b_hole])
    #     sorted_indices = np.argsort(distances)
    #     nearest_points_sorted = segment_nearest_points_b_hole[sorted_indices]

    #     points_to_remove = nearest_points_sorted[760:900]

    #     indices_to_remove = []
    #     for point in points_to_remove:
    #         # Find the index of the point in point_cloud
    #         index = np.where(np.all(point_cloud == point, axis=1))[0]
    #         if index.size > 0:
    #             indices_to_remove.extend(index)

    #     # Now remove these indices from point_cloud
    #     point_cloud_filtered = np.delete(point_cloud, indices_to_remove, axis=0)
        
    #     nearest_points_sorted[760:900, 2] -= random.uniform(0.05, 0.15)
    #     hole = nearest_points_sorted[760:900]

    #     point_cloud = np.vstack((point_cloud_filtered, hole))

    # p34_1_intersection_point, p34_1_direction_vector =  get_perpendicular(p34_1, p1, p2, p3, p4)
    # p34_5_intersection_point, p34_5_direction_vector = get_perpendicular(p34_5, p1, p2, p3, p4)
    # p34_9_intersection_point, p34_9_direction_vector =  get_perpendicular(p34_9, p1, p2, p3, p4)

    # segments_p1_p2[1] = p34_1_intersection_point
    # segments_p1_p2[5] = p34_5_intersection_point
    # segments_p1_p2[9] = p34_9_intersection_point

    # g34_1_intersection_point, g34_1_direction_vector =  get_perpendicular(g34_1, g1, g2, g3, g4)
    # g34_5_intersection_point, g34_5_direction_vector = get_perpendicular(g34_5, g1, g2, g3, g4)
    # g34_9_intersection_point, g34_9_direction_vector =  get_perpendicular(g34_9, g1, g2, g3, g4)

    # segments_g1_g2[1] = g34_1_intersection_point
    # segments_g1_g2[5] = g34_5_intersection_point
    # segments_g1_g2[9] = g34_9_intersection_point

    # Ramp Slope
    for k, i in enumerate(selected_segments_ramp):
        temp_nearest_points_a = nearest_points(segments_p1_p2[i], segments_p4_p3[i], point_cloud, gutter=False)
        nearest_points_a = np.vstack((nearest_points_a, temp_nearest_points_a))

        X = temp_nearest_points_a[:, :2]
        y = temp_nearest_points_a[:, 2]

        if activated_fit_line_with_pca_threshold:
            pca, cul_mask = fit_line_with_pca_threshold(X, y)
        if activated_fit_line_with_pca_threshold_z:
            pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_a_masked = temp_nearest_points_a[cul_mask]
        nearest_points_a_masked = np.vstack((nearest_points_a_masked, temp_nearest_points_a_masked))

        slope = calculate_slope_pca(pca)
        a_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p1_p2 = p2[:2] - p1[:2]
        d_p3_p4 = p4[:2] - p3[:2]
        intersection1_xy = find_intersection(p1[:2], d_p1_p2, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(p3[:2], d_p3_p4, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        aline_points.append([intersection_point_1, intersection_point_2])


    # Ramp Cross Slope
    for k, i in enumerate(selected_segments_ramp):
        temp_nearest_points_b = nearest_points(segments_p1_p4[i], segments_p2_p3[i], point_cloud, gutter=False)
        nearest_points_b = np.vstack((nearest_points_b, temp_nearest_points_b))

        X = temp_nearest_points_b[:, :2]
        y = temp_nearest_points_b[:, 2]
        if activated_fit_line_with_pca_threshold:
            pca, cul_mask = fit_line_with_pca_threshold(X, y)
        if activated_fit_line_with_pca_threshold_z:
            pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_b_masked = temp_nearest_points_b[cul_mask]
        nearest_points_b_masked = np.vstack((nearest_points_b_masked, temp_nearest_points_b_masked))

        slope = calculate_slope_pca(pca)
        b_calc.append(slope)

        pca_mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p4_p1 = p1[:2] - p4[:2]
        d_p2_p3 = p3[:2] - p2[:2]

        intersection1_xy = find_intersection(p4[:2], d_p4_p1, pca_mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(p2[:2], d_p2_p3, pca_mean[:2], pca_direction_xy)

        line_point = np.array(pca_mean)
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        sorted_indices = np.argsort(np.array([distance.euclidean(intersection_point_1, point) for point in temp_nearest_points_b_masked]))
        nearest_points_sorted = temp_nearest_points_b_masked[sorted_indices]
        projections = project_points_to_line(nearest_points_sorted, line_point, pca.components_[0])
        projections = np.vstack((intersection_point_1, projections))
        projections = np.vstack((projections, intersection_point_2))

        gaps = find_gaps_in_projections(projections)

        if len(gaps) > 0: # gap exist
            end_to_hole = calculate_3d_distance(intersection_point_1, projections[gaps[0][0]])
            hole_to_end = calculate_3d_distance(projections[gaps[0][1]], intersection_point_2)
            bline_points.append([intersection_point_1, projections[gaps[0][0]], intersection_point_2, projections[gaps[0][1]]])
            c_calc.append(max(end_to_hole, hole_to_end))
        else:
            bline_points.append([intersection_point_1, intersection_point_2])
            c_calc.append(calculate_3d_distance(intersection_point_1, intersection_point_2))

    # Left Flare Slope
    for k, i in enumerate(selected_segments_flare):
        temp_nearest_points_d = nearest_points(segments_p1_fl[i], segments_p1_p4[i], point_cloud, gutter=False)
        nearest_points_d = np.vstack((nearest_points_d, temp_nearest_points_d))

        X = temp_nearest_points_d[:, :2]
        y = temp_nearest_points_d[:, 2]
        if activated_fit_line_with_pca_threshold:
            pca, cul_mask = fit_line_with_pca_threshold(X, y)
        if activated_fit_line_with_pca_threshold_z:
            pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_d_masked = temp_nearest_points_d[cul_mask]
        nearest_points_d_masked = np.vstack((nearest_points_d_masked, temp_nearest_points_d_masked))

        slope = calculate_slope_pca(pca)
        d_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p1_p4 = p4[:2] - p1[:2]
        d_fl_p1 = p1[:2] - fl[:2]
        intersection1_xy = find_intersection(p1[:2], d_p1_p4, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(fl[:2], d_fl_p1, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        dline_points.append([intersection_point_1, intersection_point_2])

    # Right Flare Slope
    for k, i in enumerate(selected_segments_flare):
        temp_nearest_points_e = nearest_points(segments_p2_fr[i], segments_p2_p3[i], point_cloud, gutter=False)
        nearest_points_e = np.vstack((nearest_points_e, temp_nearest_points_e))

        X = temp_nearest_points_e[:, :2]
        y = temp_nearest_points_e[:, 2]

        if activated_fit_line_with_pca_threshold:
            pca, cul_mask = fit_line_with_pca_threshold(X, y)
        if activated_fit_line_with_pca_threshold_z:
            pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_e_masked = temp_nearest_points_e[cul_mask]
        nearest_points_e_masked = np.vstack((nearest_points_e_masked, temp_nearest_points_e_masked))

        slope = calculate_slope_pca(pca)
        e_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p2_fr = fr[:2] - p2[:2]
        d_p3_p2 = p2[:2] - p3[:2]
        intersection1_xy = find_intersection(p2[:2], d_p2_fr, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(p3[:2], d_p3_p2, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        eline_points.append([intersection_point_1, intersection_point_2])

    
    # Gutter Cross Slope
    for k, i in enumerate(selected_segments_gutter):
        temp_nearest_points_f = nearest_points(segments_g1_g4[i], segments_g2_g3[i], point_cloud, gutter=False)
        nearest_points_f = np.vstack((nearest_points_f, temp_nearest_points_f))

        X = temp_nearest_points_f[:, :2]
        y = temp_nearest_points_f[:, 2]

        if activated_fit_line_with_pca_threshold:
            pca, cul_mask = fit_line_with_pca_threshold(X, y)
        if activated_fit_line_with_pca_threshold_z:
            pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_f_masked = temp_nearest_points_f[cul_mask]
        nearest_points_f_masked = np.vstack((nearest_points_f_masked, temp_nearest_points_f_masked))

        slope = calculate_slope_pca(pca)
        f_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]

        
        d_g4_g1 = g1[:2] - g4[:2]
        d_g2_g3 = g3[:2] - g2[:2]
        intersection1_xy = find_intersection(g4[:2], d_g4_g1, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(g2[:2], d_g2_g3, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        fline_points.append([intersection_point_1, intersection_point_2])

    
    # Gutter Cross Slope
    for k, i in enumerate(selected_segments_ramp):
        temp_nearest_points_g = nearest_points(segments_g1_g2[i], segments_g4_g3[i], point_cloud, gutter=True)
        nearest_points_g = np.vstack((nearest_points_g, temp_nearest_points_g))

        X = temp_nearest_points_g[:, :2]
        y = temp_nearest_points_g[:, 2]

        if activated_fit_line_with_pca_threshold:
            pca, cul_mask = fit_line_with_pca_threshold(X, y)
        if activated_fit_line_with_pca_threshold_z:
            pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_g_masked = temp_nearest_points_g[cul_mask]
        nearest_points_g_masked = np.vstack((nearest_points_g_masked, temp_nearest_points_g_masked))

        slope = calculate_slope_pca(pca)
        g_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]

        
        d_g1_g2 = g2[:2] - g1[:2]
        d_g3_g4 = g4[:2] - g3[:2]
        intersection1_xy = find_intersection(g1[:2], d_g1_g2, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(g3[:2], d_g3_g4, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        gline_points.append([intersection_point_1, intersection_point_2])




    # VISUALIZATION
            
    trace_point_cloud = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='grey', opacity=0.4),
        name='Point Cloud'
    )

    ramp_x = [p1[0], p2[0], None, p2[0], p3[0], None, p3[0], p4[0], None, p4[0], p1[0]]
    ramp_y = [p1[1], p2[1], None, p2[1], p3[1], None, p3[1], p4[1], None, p4[1], p1[1]]
    ramp_z = [p1[2], p2[2], None, p2[2], p3[2], None, p3[2], p4[2], None, p4[2], p1[2]]

    # Creating a single trace for all segments
    trace_ramp_outline = go.Scatter3d(x=ramp_x, y=ramp_y, z=ramp_z, mode='lines', line=dict(width=2, color='black'), name='Ramp Outline')

    gutter_x = [g1[0], g2[0], None, g2[0], g3[0], None, g3[0], g4[0], None, g4[0], g1[0]]
    gutter_y = [g1[1], g2[1], None, g2[1], g3[1], None, g3[1], g4[1], None, g4[1], g1[1]]
    gutter_z = [g1[2], g2[2], None, g2[2], g3[2], None, g3[2], g4[2], None, g4[2], g1[2]]

    # Creating a single trace for all segments
    trace_gutter_outline = go.Scatter3d(x=gutter_x, y=gutter_y, z=gutter_z, mode='lines', line=dict(width=2, color='black'), name='Gutter Outline')

    flare_left_x = [p1[0], fl[0], None, p4[0], fl[0]]
    flare_left_y = [p1[1], fl[1], None, p4[1], fl[1]]
    flare_left_z = [p1[2], fl[2], None, p4[2], fl[2]]

    trace_flare_left_outline = go.Scatter3d(x=flare_left_x, y=flare_left_y, z=flare_left_z, mode='lines', line=dict(width=2, color='black'), name='Left Flare Outline')

    flare_right_x = [p2[0], fr[0], None, p3[0], fr[0]]
    flare_right_y = [p2[1], fr[1], None, p3[1], fr[1]]
    flare_right_z = [p2[2], fr[2], None, p3[2], fr[2]]

    trace_flare_right_outline = go.Scatter3d(x=flare_right_x, y=flare_right_y, z=flare_right_z, mode='lines', line=dict(width=2, color='black'), name='Right Flare Outline')

    trace_a_points = go.Scatter3d(
        x=nearest_points_a_masked[:, 0],
        y=nearest_points_a_masked[:, 1],
        z=nearest_points_a_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='green', opacity=0.4),
        name='Ramp slope points (A)'
    )

    for i, pts in enumerate(aline_points):
        trace_a_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='green', width=4),
            text=f'A{i+1}',
            name=f'Ramp slope (A{i+1})'
        )
        trace_a_lines.append(trace_a_line)

    trace_a = [trace_a_points] + trace_a_lines

    trace_b_points = go.Scatter3d(
        x=nearest_points_b_masked[:, 0],
        y=nearest_points_b_masked[:, 1],
        z=nearest_points_b_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='red', opacity=0.5),
        name='Ramp cross slope points (B)'
    )

    for i, pts in enumerate(bline_points):
        if len(pts) == 2:
            trace_b_line = go.Scatter3d(
                x=[pts[0][0], pts[1][0]],
                y=[pts[0][1], pts[1][1]],
                z=[pts[0][2], pts[1][2]],
                mode='lines+text',
                line=dict(color='red', width=4),
                text=f'B{i+1}',
                name=f'Ramp cross slope (B{i+1})'
            )
            trace_b_lines.append(trace_b_line)
        else:
            trace_b_line = go.Scatter3d(
                x=[pts[0][0], pts[1][0], None, pts[2][0], pts[3][0]],
                y=[pts[0][1], pts[1][1], None, pts[2][1], pts[3][1]],
                z=[pts[0][2], pts[1][2], None, pts[2][2], pts[3][2]],
                mode='lines+text',
                line=dict(color='red', width=4),
                text=f'B{i+1}',
                name=f'Ramp cross slope (B{i+1})'
            )
            trace_b_lines.append(trace_b_line)

    trace_b = [trace_b_points] + trace_b_lines

    trace_d_points = go.Scatter3d(
        x=nearest_points_d_masked[:, 0],
        y=nearest_points_d_masked[:, 1],
        z=nearest_points_d_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='brown', opacity=0.5),
        name='Left flare slope points (D)'
    )

    for i, pts in enumerate(dline_points):
        trace_d_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='brown', width=4),
            text=f'D{i+1}',
            name=f'Left flare slope (D{i+1})'
        )
        trace_d_lines.append(trace_d_line)

    trace_d = [trace_d_points] + trace_d_lines

    trace_e_points = go.Scatter3d(
        x=nearest_points_e_masked[:, 0],
        y=nearest_points_e_masked[:, 1],
        z=nearest_points_e_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='brown', opacity=0.5),
        name='Right flare slope points (E)'
    )

    for i, pts in enumerate(eline_points):
        trace_e_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='brown', width=4),
            text=f'E{i+1}',
            name=f'Right flare slope (E{i+1})'
        )
        trace_e_lines.append(trace_e_line)

    trace_e = [trace_e_points] + trace_e_lines

    trace_f_points = go.Scatter3d(
        x=nearest_points_f_masked[:, 0],
        y=nearest_points_f_masked[:, 1],
        z=nearest_points_f_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='orange', opacity=0.5),
        name='Gutter slope points (F)'
    )

    for i, pts in enumerate(fline_points):
        trace_f_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='orange', width=4),
            text=f'F{i+1}',
            name=f'Gutter slope (F{i+1})'
        )
        trace_f_lines.append(trace_f_line)

    trace_f = [trace_f_points] + trace_f_lines

    trace_g_points = go.Scatter3d(
        x=nearest_points_g_masked[:, 0],
        y=nearest_points_g_masked[:, 1],
        z=nearest_points_g_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='blue', opacity=0.5),
        name='Gutter slope points (G)'
    )

    for i, pts in enumerate(gline_points):
        trace_g_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='blue', width=4),
            text=f'G{i+1}',
            name=f'Gutter slope (G{i+1})'
        )
        trace_g_lines.append(trace_g_line)

    trace_g = [trace_g_points] + trace_g_lines

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.1),
            camera=dict(
                eye=dict(x=0.3, y=0.6, z=0.3),
            )
        ),
        title='3D Visualization of Point Cloud and Selected Points'
    )

    traces = [
        trace_point_cloud, 
        trace_ramp_outline,
        trace_flare_left_outline,
        trace_flare_right_outline,
        trace_gutter_outline,
    ]

    traces += trace_a
    traces += trace_b
    traces += trace_d
    traces += trace_e
    traces += trace_f
    traces += trace_g

    # fig = go.Figure(data=traces, layout=layout)
    # fig.show()


    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        specs=[[{'type': 'scatter3d', 'rowspan': 1}], [{'type': 'table'}]]
    )

    for trace in traces:
        fig.add_trace(trace, row=1, col=1)

    fig.update_layout(layout) 

    headers = [
        'Ramp Slope (x.x%)', 'Ramp Cross Slope (x.x%)', 'Ramp Width (inches)', 
         'Left Flare Slope at Back of Curb (x.x%)', 'Right Flare Slope at Back of Curb (x.x%)', 'Gutter Slope (x.x%)', 'Gutter Cross Slope (x.x%)'
        ]
    rows = [
        [f'A1: {a_calc[0]}%', f'A2: {a_calc[1]}%', f'A3: {a_calc[2]}%', f'8.3% or less?', f'{yes_no(max(a_calc)<=8.3)}', 'ADA Standard 8.3% max'],
        [f'B1: {b_calc[0]}%', f'B2: {b_calc[1]}%', f'B3: {b_calc[2]}%', f'2.0% or less?', f'{yes_no(max(b_calc)<=2.0)}', 'ADA Standard 2.0% max'],
        [f'C1: {c_calc[0]}"', f'C2: {c_calc[1]}"', f'C3: {c_calc[2]}"', f'48" or greater?', f'{yes_no(min(c_calc)>=48)}', 'ADA Standard 48" min'],
        [f'D1: {d_calc[0]}%', f'D1: {d_calc[1]}%', '', f'10.0% or less?', f'{yes_no(max(d_calc)<=10)}', 'ADA Standard 10.0% max'], 
        [f'E1: {e_calc[0]}%', f'E1: {e_calc[1]}%', '', f'10.0% or less?', f'{yes_no(max(e_calc)<=10)}', 'ADA Standard 10.0% max'],
        [f'F1: {f_calc[0]}%', f'F2: {f_calc[1]}%', '', f'2.0% or less?', f'{yes_no(max(f_calc)<=2)}', 'ADA Standard 2.0% max'],
        [f'G1: {g_calc[0]}%', f'G2: {g_calc[1]}%', f'G3: {g_calc[2]}%', f'5.0% or less?', f'{yes_no(max(g_calc)<=5)}', 'ADA Standard 5.0% max'],


    #     # Add more rows as needed
    ]


    # Add the table to the second column
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color='lightgrey', align='left'),
            cells=dict(values=[rows[i] for i in range(len(headers))],fill_color='whitesmoke', align='left')
        ),
        row=2, col=1
    )
    fig.update_layout(title=f"ADA CURB RAMP FIELD MEASUREMENTS ({las_file_path.split('/')[-1]})", showlegend=True)

    fig.show()
