import numpy as np
from validation import run_sklearn_classification, run_sklearn_regression

from clustering import (transform_data,
                        kmeans,
                        plot_clusters,
                        elbow_method,
                        visualize_elbow)
from data import (load_data,
                  add_derived_features,
                  analyze_data,
                  add_demand_category)
from evaluation import (get_folds,
                        f1_score,
                        rmse,
                        cross_validation_scores,
                        plot_cv_results, find_best_k)
from knn import ClassificationKNN, RegressionKNN

np.random.seed(2)


def run_clustering(df, features):
    print("\nK-means clustering")
    data = transform_data(df, features)

    #use elbow method to inspect a reasonable range of k values
    k_values = list(range(1, 11))
    inertia_values = elbow_method(data, k_values)
    visualize_elbow(k_values, inertia_values, "outputs/elbow_plot.png")

    k_list = [2, 3, 5]

    #run K-means for selected values of k
    for k in k_list:
        labels, centroids = kmeans(data, k)

        print(f"k = {k}")
        print(np.array_str(centroids, precision=3, suppress_small=True))

        plot_clusters(data, labels, centroids, features, f"outputs/clusters_k_{k}.png")


def run_classification(data, folds, features):
    print("\nKNN Classification with features:")
    for f in features:
        print(f" - {f}")

    data = add_demand_category(data)

    # Define targets for classification
    X = data[features].values
    y = data['demand_class'].values

    k_list = [3, 5, 11, 25, 51, 75, 101]
    scores_mean = []

    # Evaluate classification performance for each k using F1 score
    for k in k_list:
        model = ClassificationKNN(k)
        scores = cross_validation_scores(model, X, y, folds, f1_score)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        scores_mean.append(mean)
        print(f"k={k} | F1-macro={mean:.4f} ± {std:.4f}")

    best_k, best_score = find_best_k(k_list, scores_mean)
    print(f"Best k={best_k} with F1-macro={best_score:.4f}")

    # Visualize classification results
    plot_cv_results(k_list, scores_mean, "F1 score", "KNN Classification", "outputs/classification.png")


def run_regression(data, folds, features):
    print("\nKNN Regression with features:")
    for f in features:
        print(f" - {f}")

    # Define target for regression
    X = data[features].values
    y = data['cnt'].values

    k_list = [3, 5, 11, 25, 51, 75, 101]
    scores_mean = []

    # Evaluate regression performance for each k using RMSE
    for k in k_list:
        model = RegressionKNN(k)
        scores = cross_validation_scores(model, X, y, folds, rmse)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        scores_mean.append(mean)
        print(f"k={k} | RMSE={mean:.4f} ± {std:.4f}")

    best_k, best_score = find_best_k(k_list, scores_mean, maximize=False)
    print(f"Best k={best_k} with RMSE={best_score:.4f}")

    # Visualize regression results
    plot_cv_results(k_list, scores_mean, "RMSE", "KNN Regression", "outputs/regression.png")


def main():
    print("Loading data and running analysis")
    df = load_data("data/london_sample_2500.csv")
    df = add_derived_features(df)
    print("\n=== DATA ANALYSIS ===")
    analyze_data(df)

    folds = get_folds()

    print("\n=== CORE EXPERIMENTS ===")
    run_clustering(df, ['cnt', 't1'])
    run_classification(df, folds, ['hour_sin', 'hour_cos', 'is_weekend_holiday', 't1', 'weather_code'])
    run_classification(df, folds, ['hour_sin', 'hour_cos', 'is_weekend_holiday', 'season_sin', 'season_cos'])

    run_regression(df, folds, ['hour_sin', 'hour_cos', 'is_weekend_holiday', 't1', 'weather_code'])
    run_regression(df, folds, ['hour_sin', 'hour_cos', 'is_weekend_holiday', 'season_sin', 'season_cos'])

    print("\n=== VALIDATION AGAINST SKLEARN ===")
    run_sklearn_classification(df, folds, ['hour_sin', 'hour_cos', 'is_weekend_holiday', 't1', 'weather_code'])
    run_sklearn_regression(df, folds, ['hour_sin', 'hour_cos', 'is_weekend_holiday', 't1', 'weather_code'])


if __name__ == "__main__":
    main()
