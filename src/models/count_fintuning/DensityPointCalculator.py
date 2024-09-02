import numpy as np
from sklearn.neighbors import NearestNeighbors

class DensityPointCalculator:
    def __init__(self, data):
        """
        Initialize the calculator with the dataset.
        Set up the nearest neighbor model to find the 20 nearest points.
        """
        self.data = data
        self.model = NearestNeighbors(n_neighbors=21)  # 20 nearest + 1 (itself because it will always be the nearest)
        self.model.fit(data)

    def choose_random_point(self):
        """
        Select a random point from the dataset.
        """
        index = np.random.randint(0, len(self.data))
        return self.data[index], index

    def find_nearest_points(self, point):
        """
        Find the 20 nearest points to a given point.
        """
        distances, indices = self.model.kneighbors([point])
        return self.data[indices[0][1:]]  # Exclude the point itself to get only the nearest points

    def select_points_with_variance(self, points, max_variance=0.03):
        """
        Select k nearest points until their variance is less than the specified max_variance.
        """
        k = 40
        for k in range(1, len(points) + 1):
            if np.var(points[:k], axis=0).mean() < max_variance:
                break
        return points[:k], k

    def compute_knn_average(self, points):
        """
        Compute the average of the selected k-nearest neighbors.
        """
        return points.mean(axis=0)

    def compute_number_of_points(self, head, tail, points, k):
        """
        Compute the number of points between the provided 'head' and 'tail' of the line.
        """
        m = self.compute_knn_average(points)
        numerator = abs(np.linalg.norm(head - tail))
        denominator = np.linalg.norm(m) * k
        num_points = numerator / denominator if denominator != 0 else 0
        return num_points

    def average_density_points(self, head, tail, trials=10):
        """
        Repeat the point selection and calculation for a number of trials and return the average number of points.
        """
        results = []
        for _ in range(trials):
            point, _ = self.choose_random_point()
            nearest_points = self.find_nearest_points(point)
            selected_points, k = self.select_points_with_variance(nearest_points)
            num_points = self.compute_number_of_points(head, tail, selected_points, k)
            results.append(num_points)
        return np.mean(results)

# Example Usage
data = np.random.rand(100, 3)  # 100 points in 5 dimensions
head = np.array([1, 1, 0])  # Example head coordinates
tail = np.array([1, 1, 10])  # Example tail coordinates
calculator = DensityPointCalculator(data)
print(f"Average Number of Points: {calculator.average_density_points(head, tail)}")

