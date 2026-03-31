import numpy as np
import matplotlib.pyplot as plt

def validate_k(data, k):
    """
    Validate the number of clusters.
    """
    n_samples = len(data)

    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k > n_samples:
        raise ValueError("k cannot be greater than the number of data points.")


def validate_data_array(data):
    """
    Validate clustering input data.
    """
    if len(data.shape) != 2:
        raise ValueError("Input data must be a 2D array.")

    if data.shape[0] == 0:
        raise ValueError("Input data cannot be empty.")

    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")


def validate_plot_features(features):
    """
    Validate features used for 2D plotting.
    """
    if len(features) != 2:
        raise ValueError("Exactly 2 features are required for cluster plotting.")


def choose_initial_centroids(data, k):
    """
    Randomly choose k distinct data points as initial centroids.
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


def transform_data(df, features):
    """
    Select features for clustering and scale them to [0, 1].
    """
    data = df[features].to_numpy(dtype=float)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # avoid division by zero for constant features

    transformed_data = (data - min_vals) / ranges

    return transformed_data


def dist(x, y):
    """
    Compute Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((x - y) ** 2))


def assign_to_clusters(data, centroids):
    """
    Assign each data point to its nearest centroid.
    """
    squared_distances = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    return np.argmin(squared_distances, axis=1)


def recompute_centroids(data, labels, k):
    """
    Recompute centroids as the mean of the points assigned to each cluster.
    If a cluster becomes empty, reinitialize its centroid randomly from the data.
    """
    new_centroids = []
    for cluster_index in range(k):
        cluster_points = data[labels == cluster_index]
        if len(cluster_points) == 0:
            random_index = np.random.choice(data.shape[0])
            centroid = data[random_index]
        else:
            centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(centroid)
    return np.array(new_centroids)


def kmeans(data, k, max_iter=100):
    """
    Run the K-means clustering algorithm until convergence
    or until reaching the maximum number of iterations.
    """
    validate_data_array(data)
    validate_k(data, k)

    centroids = choose_initial_centroids(data, k)

    for _ in range(max_iter):
        labels = assign_to_clusters(data, centroids)
        new_centroids = recompute_centroids(data, labels, k)

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


def plot_clusters(data, labels, centroids, features, save_path=None):
    """
    Plot clustered points and their centroids, then save the figure.
    """
    validate_plot_features(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.7)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='X',
        s=200,
        color='red',
        label='Centroids'
    )

    plt.title("K-means Clustering")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()




def compute_inertia(data, labels, centroids):
    """
    Compute the within-cluster sum of squares (WCSS), also called inertia.
    """
    return np.sum((data - centroids[labels]) ** 2)


def elbow_method(data, k_values):
    """
    Run K-means for multiple values of k and compute inertia for each one.
    """
    inertia_values = []

    for k in k_values:
        labels, centroids = kmeans(data, k)
        inertia = compute_inertia(data, labels, centroids)
        inertia_values.append(inertia)

    return inertia_values


def visualize_elbow(k_values, inertia_values, path):
    """
    Plot the elbow curve and save it to a file.
    """
    plt.figure()
    plt.plot(k_values, inertia_values, marker='o')
    plt.title("Elbow Method for K-means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.grid(True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
