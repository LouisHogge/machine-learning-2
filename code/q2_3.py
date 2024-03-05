"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from data import load_superconduct
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

RIDGE = "ridge_regressor"
KNN = "knn_regressor"
DT = "dt_regressor"

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
N = 500
RAND_STATE = 42

# max value of each hyper-parameters
DEPTHS = 30
NEIGHBORS = 20
ALPHAS = 50


def model_choice(model_used, param, rand_value):
    """
    Selects and configures a machine learning model based on specified parameters.

    Parameters:
    - model_used (constant): Identifier for the model to be used (RIDGE, KNN, or DT).
    - param (int): Parameter for model configuration. Interpreted as alpha for Ridge (starting at 0),
      n_neighbors for KNN (starting at 1), or max_depth for DT (starting at 1).
    - rand_value (int): Multiplier for setting the random state in models that support it.

    Returns:
    - model: The configured model instance.
    """

    if model_used == RIDGE:
        # alpha starts at 0
        model = Ridge(alpha=param-1, random_state=(RAND_STATE*rand_value))

    if model_used == KNN:
        # n_neighbors starts at 1
        # no random_state= for KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=param)

    if model_used == DT:
        # depth starts at 1
        model = DecisionTreeRegressor(
            max_depth=param, random_state=(RAND_STATE*rand_value))

    return model


def plot_model_q2_3(model_used, complex_param, param, expect_error, var):
    """
    Plots and displays a graph comparing expected error, variance, and bias^2 + residual error 
    for a model over a range of complexity parameter values.

    Parameters:
    - model_used (constant): Identifier for the model used (RIDGE, KNN, or DT).
    - complex_param (str): The name of the complexity parameter being varied.
    - param (int): The maximum value of the complexity parameter to plot.
    - expect_error (list): List of expected error values corresponding to each complexity parameter value.
    - var (list): List of variance values corresponding to each complexity parameter value.

    Returns:
    - None
    """

    if model_used == RIDGE:
        x_complex = np.arange(0, param, 1)
    else:
        x_complex = np.arange(1, param + 1, 1)

    plt.plot(x_complex, expect_error, label="Expected error")
    plt.plot(x_complex, var, label="Variance")
    plt.plot(x_complex, expect_error - var,
             label=r"$\mathrm{bias^2 + redidual \ error}$")

    plt.legend(loc='best')
    plt.xlabel("Complexity " + complex_param)
    plt.ylabel("Error")
    # plt.title("Error of " + model_used + " related to complexity")
    plt.savefig('2.3_' + model_used + '.pdf')
    plt.show()


if __name__ == "__main__":

    X, y = load_superconduct()

    # separation dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RAND_STATE)

    # list of model to test
    models = [RIDGE, KNN, DT]
    # list main complexity param
    complex_params = [r'($\mathrm{\alpha}$)',
                      r'($\mathrm{k}$)', r'($\mathrm{max\_depth}$)']

    # list of max value of hyper-parameters
    hyper_params = [DEPTHS, NEIGHBORS, ALPHAS]

    # nb learning sample
    number_LS = 50

    # true error =
    true_error = np.empty(number_LS)
    predictions = np.empty((number_LS, X_test.shape[0]))

    for index, model_used in enumerate(models):

        var = np.empty(hyper_params[index])
        expect_error = np.empty(hyper_params[index])

        # for value hyper-param from 1 -> max_value
        for param in range(1, hyper_params[index] + 1):

            model = model_choice(model_used, param, rand_value=param)

            # for on multiple LS to average
            for ls in range(number_LS):

                # shuffle training set and take the N(=500) first to train
                X_t, y_t = shuffle(
                    X_train, y_train, random_state=(RAND_STATE*ls))
                model.fit(X_t[:N], y_t[:N])

                predictions[ls] = model.predict(X_test)
                true_error[ls] = mean_squared_error(
                    y_true=y_test, y_pred=predictions[ls])  # Ey|x{(y - y_hat(x))^2}

            # var and except_error over all LS
            var[param-1] = np.mean(np.var(predictions, axis=0))
            expect_error[param-1] = np.mean(np.mean(true_error))

        plot_model_q2_3(
            model_used, complex_params[index], param, expect_error, var)
