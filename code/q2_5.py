import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from data import load_superconduct


def plot_comparison(no_bagging_result, bagging_result, model_name):
    """
    Plots and saves a bar chart comparing bias and variance for a model with and without bagging.

    Parameters:
    - no_bagging_result (tuple): Bias and variance for the model without bagging.
    - bagging_result (tuple): Bias and variance for the model with bagging.
    - model_name (str): Name of the model.

    Returns:
    - None
    """

    metrics = ['bias', 'variance']
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar([x for x in range(len(metrics))],
           [no_bagging_result[j] for j in range(len(metrics))],
           width=0.4, label=f'{model_name} without Bagging')

    ax.bar([x + 0.4 for x in range(len(metrics))],
           [bagging_result[j] for j in range(len(metrics))],
           width=0.4, label=f'{model_name} with Bagging')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(f'{model_name} - Comparison With and Without Bagging')
    ax.set_xticks([r + 0.2 for r in range(len(metrics))])
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.savefig(f'2_5_{model_name}_comparison.pdf', bbox_inches='tight')
    plt.close()


def calculate_bias_variance(y_true, y_pred):
    """
    Calculates the bias and variance from true values and predictions.

    Parameters:
    - y_true (array-like): True values of the target variable.
    - y_pred (array-like): Predicted values of the target variable.

    Returns:
    - bias: The calculated bias.
    - variance: The calculated Variance.
    """

    mean_predictions = np.mean(y_pred, axis=0)
    bias = np.mean((mean_predictions - y_true) ** 2)
    variance = np.mean(np.var(y_pred, axis=0))
    return bias, variance


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Fits a model using the training data and evaluates its performance on the test data.

    Parameters:
    - model (model object): The machine learning model to be trained and evaluated.
    - X_train (array-like): Training features.
    - y_train (array-like): Training target variable.
    - X_test (array-like): Test features.
    - y_test (array-like): Test target variable.

    Returns:
    - bias: The calculated bias for the model predictions.
    - variance: The calculated variance for the model predictions.
    """

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    bias = np.mean((predictions - y_test) ** 2)
    variance = np.var(predictions)
    return bias, variance


def evaluate_bagging(model, X_train, y_train, X_test, y_test, n_estimators=10):
    """
    Evaluates a model with bagging on the given dataset and calculates the bias and variance.

    Parameters:
    - model (model object): The base machine learning model to apply bagging to.
    - X_train (array-like): Training features.
    - y_train (array-like): Training target variable.
    - X_test (array-like): Test features.
    - y_test (array-like): Test target variable.
    - n_estimators (int, optional): The number of base estimators in the bagging ensemble. Defaults to 10.

    Returns:
    - bias: The calculated bias for the bagging ensemble.
    - variance: The calculated variance for the bagging ensemble.
    """

    bagging_model = BaggingRegressor(
        estimator=model, n_estimators=n_estimators, random_state=42)
    bagging_model.fit(X_train, y_train)
    predictions = np.zeros((n_estimators, len(y_test)))
    for i in range(n_estimators):
        predictions[i, :] = bagging_model.estimators_[i].predict(X_test)

    bias, variance = calculate_bias_variance(y_test, predictions)
    return bias, variance


if __name__ == "__main__":
    X, y = load_superconduct()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    ridge_model = Ridge(alpha=30.0)
    knn_model = KNeighborsRegressor(n_neighbors=1)
    tree_model = DecisionTreeRegressor(max_depth=None)

    ridge_bagging_results = evaluate_bagging(
        ridge_model, X_train, y_train, X_test, y_test)
    knn_bagging_results = evaluate_bagging(
        knn_model, X_train, y_train, X_test, y_test)
    tree_bagging_results = evaluate_bagging(
        tree_model, X_train, y_train, X_test, y_test)

    ridge_no_bagging_results = evaluate_model(
        ridge_model, X_train, y_train, X_test, y_test)
    knn_no_bagging_results = evaluate_model(
        knn_model, X_train, y_train, X_test, y_test)
    tree_no_bagging_results = evaluate_model(
        tree_model, X_train, y_train, X_test, y_test)

    print(
        f"Ridge Regression without Bagging - Bias: {ridge_no_bagging_results[0]:.4f}, Variance: {ridge_no_bagging_results[1]:.4f}")
    print(
        f"Ridge Regression with Bagging - Bias: {ridge_bagging_results[0]:.4f}, Variance: {ridge_bagging_results[1]:.4f}\n")

    print(
        f"kNN without Bagging - Bias: {knn_no_bagging_results[0]:.4f}, Variance: {knn_no_bagging_results[1]:.4f}")
    print(
        f"kNN with Bagging - Bias: {knn_bagging_results[0]:.4f}, Variance: {knn_bagging_results[1]:.4f}\n")

    print(
        f"Regression Tree without Bagging - Bias: {tree_no_bagging_results[0]:.4f}, Variance: {tree_no_bagging_results[1]:.4f}")
    print(
        f"Regression Tree with Bagging - Bias: {tree_bagging_results[0]:.4f}, Variance: {tree_bagging_results[1]:.4f}\n")

    plot_comparison(ridge_no_bagging_results,
                    ridge_bagging_results, 'Ridge Regression')
    plot_comparison(knn_no_bagging_results, knn_bagging_results, 'kNN')
    plot_comparison(tree_no_bagging_results,
                    tree_bagging_results, 'Regression Tree')
