import numpy as np
from statistics import mode
from abc import abstractmethod, ABC


class StandardScaler:
    def __init__(self):
        """Initialize scaler parameters."""
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Learn per-feature mean and standard deviation from the training data."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1   # avoid division by zero for constant features

    def transform(self, X):
        """Standardize X using the previously learned statistics."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transform.")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit the scaler on X and return the standardized data."""
        self.fit(X)
        return self.transform(X)


class KNN(ABC):
    def __init__(self, k):
        """Store k and initialize the scaler and training data placeholders."""
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        self.k = k
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fit the scaler on the training set and store the scaled training data."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")

        if len(X_train) == 0:
            raise ValueError("X_train cannot be empty.")

        if len(y_train) == 0:
            raise ValueError("y_train cannot be empty.")

        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples.")

        if self.k > len(X_train):
            raise ValueError("k cannot be greater than the number of training samples.")

        if np.isnan(X_train).any():
            raise ValueError("X_train contains NaN values.")

        if np.isnan(y_train).any():
            raise ValueError("y_train contains NaN values.")

        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train

    @abstractmethod
    def predict(self, X_test):
        """Predict outputs for X_test."""
        pass

    def neighbors_indices(self, x):
        """Return the indices of the k nearest neighbors of x in the training set."""
        distances = np.sum((self.X_train - x) ** 2, axis=1)
        nearest_indices = np.argpartition(distances, self.k)[:self.k]
        return nearest_indices

    @staticmethod
    def dist(x1, x2):
        """Compute Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x1 - x2) ** 2))


class ClassificationKNN(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X_test):
        """Predict class labels using majority vote among the k nearest neighbors."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before calling predict.")

        y_pred = []
        X_test_scaled = self.scaler.transform(X_test)

        for x in X_test_scaled:
            indices = self.neighbors_indices(x)
            neighbor_labels = self.y_train[indices]  # found y_train on these indices
            y_pred.append(mode(neighbor_labels))
        return np.array(y_pred)


class RegressionKNN(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X_test):
        """Predict target values using the mean of the k nearest neighbors."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before calling predict.")

        X_test_scaled  = self.scaler.transform(X_test)
        y_pred = []
        for x in X_test_scaled:
            indices = self.neighbors_indices(x)
            neighbor_targets = self.y_train[indices]  # found y_train on these indices
            y_pred.append(np.mean(neighbor_targets))
        return np.array(y_pred)
