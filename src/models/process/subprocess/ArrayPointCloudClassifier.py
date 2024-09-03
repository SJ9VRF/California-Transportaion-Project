import numpy as np
from sklearn import svm
import plotly.graph_objects as go
import plotly.io as pio

class ArrayPointCloudClassifier:
    def __init__(self):
        pass

    def train_classifier(self, X, y):
        self.clf = svm.SVC(kernel='linear')
        self.clf.fit(X, y)
        return self.clf.coef_[0], self.clf.intercept_[0]

    def plot_decision_boundary(self, X, X_a, X_b, w, b):
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

# Example usage
if __name__ == "__main__":
    # Assume X_a and X_b are provided as numpy arrays
    X_a = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
    X_b = np.array([[3.0, 2.0, 1.0], [3.5, 2.5, 1.5], [4.0, 3.0, 2.0]])

    classifier = ArrayPointCloudClassifier()

    y_a = np.zeros(X_a.shape[0])
    y_b = np.ones(X_b.shape[0])

    X = np.vstack((X_a, X_b))
    y = np.hstack((y_a, y_b))

    w, b = classifier.train_classifier(X, y)
    classifier.plot_decision_boundary(X, X_a, X_b, w, b)

