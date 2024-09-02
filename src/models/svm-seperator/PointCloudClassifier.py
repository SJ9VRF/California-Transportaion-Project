import numpy as np
from sklearn import svm
import plotly.graph_objects as go
import plotly.io as pio
import visualise

class PointCloudClassifier:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_point_cloud_data(self, classification_value):
        return visualise.get_las_point_cloud_with_classification(self.file_path, classification_value)

    def train_classifier(self, X, y):
        self.clf = svm.SVC(kernel='linear')
        self.clf.fit(X, y)
        return self.clf.coef_[0], self.clf.intercept_[0]

    def plot_decision_boundary(self, X_a, X_b, w, b):
        xx, yy = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50),
                             np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 50))
        zz = (-w[0] * xx - w[1] * yy - b) / w[2]

        scatter_a = go.Scatter3d(
            x=X_a[:, 0], y=X_a[:, 1], z=X_a[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Group A'
        )
        scatter_b = go.Scatter3d(
            x=X_b[:, 0], y=X_b[:, 1], z=X_b[:, 2],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='Group B'
        )
        surface = go.Surface(
            x=xx, y=yy, z=zz,
            colorscale='Viridis',
            opacity=0.5,
            showscale=False
        )
        fig = go.Figure(data=[scatter_a, scatter_b, surface])
        fig.update_layout(
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3'
            ),
            title='SVM Decision Boundary with Point Clouds'
        )
        pio.write_html(fig, file='svm_decision_boundary.html', auto_open=False)
        fig.show()

# Usage Example
file_path = visualise.file_paths[0]
classifier = PointCloudClassifier(file_path)

classification_value_a = 3
classification_value_b = 2

X_a = classifier.load_point_cloud_data(classification_value_a)
X_b = classifier.load_point_cloud_data(classification_value_b)

y_a = np.zeros(X_a.shape[0])
y_b = np.ones(X_b.shape[0])

X = np.vstack((X_a, X_b))
y = np.hstack((y_a, y_b))

w, b = classifier.train_classifier(X, y)
classifier.plot_decision_boundary(X_a, X_b, w, b)
