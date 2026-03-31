import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def get_folds():
    """
    Create a fixed 5-fold cross-validation split.
    """
    return KFold(n_splits=5, shuffle=True, random_state=42)

def validate_features_and_target(X, y):
    """
    Validate feature matrix X and target vector y.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.size == 0:
        raise ValueError("X cannot be empty.")

    if y.size == 0:
        raise ValueError("y cannot be empty.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    if np.isnan(X).any():
        raise ValueError("X contains NaN values.")

    if np.isnan(y).any():
        raise ValueError("y contains NaN values.")


def validate_metric(metric):
    """
    Validate the evaluation metric.
    """
    if not callable(metric):
        raise TypeError("metric must be a callable function.")


def validate_plot_inputs(k_list, scores):
    """
    Validate inputs for cross-validation plotting.
    """
    if len(k_list) == 0:
        raise ValueError("k_list cannot be empty.")

    if len(scores) == 0:
        raise ValueError("scores cannot be empty.")

    if len(k_list) != len(scores):
        raise ValueError("k_list and scores must have the same length.")


def f1_score(y_true, y_pred):
    """
    Compute macro F1 score for multiclass classification.
    """
    classes = np.unique(y_true)
    f1_scores = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        if tp == 0:
            f1_scores.append(0)
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return np.mean(f1_scores)


def rmse(y_true, y_pred):
    """
    Compute the root mean squared error for a regression task.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def cross_validation_scores(model, X, y, folds, metric):
    """
    Perform cross-validation and return metric scores for each fold.
    """
    validate_features_and_target(X, y)
    validate_metric(metric)


    scores = []

    for train_index, val_index in folds.split(X):
        # Split data into train and validation sets
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]

        # Train model
        model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)

        # Evaluate
        scores.append(metric(y_val, y_val_pred))

    return scores


def find_best_k(k_list, scores, maximize=True):
    """
    Select the best k based on evaluation scores.
    """
    if maximize:
        best_index = np.argmax(scores)
    else:
        best_index = np.argmin(scores)

    return k_list[best_index], scores[best_index]


def plot_cv_results(k_list, scores, metric_name, title, path):
    """
    Plot cross-validation performance as a function of k.
    """
    validate_plot_inputs(k_list, scores)

    plt.figure(figsize=(8, 8))
    plt.plot(k_list, scores, marker='o', linewidth=2)

    plt.xlabel("Number of neighbors (k)")
    plt.ylabel(metric_name)
    plt.title(title)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, bbox_inches="tight")

    plt.close()
