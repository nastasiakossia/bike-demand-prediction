import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import add_demand_category


def run_sklearn_classification(data, folds, features):
    """
    Run sklearn KNN classification as a baseline check
    against the custom implementation.
    """
    print("\nSklearn KNN Classification")

    data = add_demand_category(data.copy())

    X = data[features].to_numpy()
    y = data["demand_class"].to_numpy()

    k_list = [3, 5, 11, 25, 51, 75, 101]
    f1_means = []
    f1_stds = []

    for k in k_list:
        fold_scores = []

        for train_index, val_index in folds.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            model = KNeighborsClassifier(
                n_neighbors=k,
                metric="euclidean",
                weights="uniform"
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            fold_scores.append(f1_score(y_val, y_pred, average="macro"))

        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores, ddof=1)

        f1_means.append(mean_f1)
        f1_stds.append(std_f1)

        print(f"k={k} | F1-macro={mean_f1:.4f} ± {std_f1:.4f}")

    best_k = k_list[np.argmax(f1_means)]
    print(f"Best k={best_k} with F1-macro={f1_means[np.argmax(f1_means)]:.4f}")

    return {
        "k_list": k_list,
        "f1_macro": f1_means,
        "f1_std": f1_stds,
        "best_k": best_k,
    }


def run_sklearn_regression(data, folds, features):
    """
    Run sklearn KNN regression as a baseline check
    against the custom implementation.
    """
    print("\nSklearn KNN Regression")

    X = data[features].to_numpy()
    y = data["cnt"].to_numpy()

    k_list = [3, 5, 11, 25, 51, 75, 101]
    rmse_means = []
    rmse_stds = []

    for k in k_list:
        fold_scores = []

        for train_index, val_index in folds.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            model = KNeighborsRegressor(
                n_neighbors=k,
                metric="euclidean",
                weights="uniform"
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_scores.append(rmse)

        mean_rmse = np.mean(fold_scores)
        std_rmse = np.std(fold_scores, ddof=1)

        rmse_means.append(mean_rmse)
        rmse_stds.append(std_rmse)

        print(f"k={k} | RMSE={mean_rmse:.4f} ± {std_rmse:.4f}")

    best_k = k_list[np.argmin(rmse_means)]
    print(f"Best k={best_k} with RMSE={rmse_means[np.argmin(rmse_means)]:.4f}")

    return {
        "k_list": k_list,
        "rmse": rmse_means,
        "rmse_std": rmse_stds,
        "best_k": best_k,
    }