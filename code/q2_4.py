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

ALPHA_FIXED = [3]
K_FIXED = [7]
DEPTHS_FIXED = [1, 2, 4, 8, None]

MIN_LS = 500
MAX_LS = 17000
STEP_LS = 500


def model_choice(model_used, param, rand_value):
    """
    Selects and configures a machine learning model based on the provided parameters.

    Parameters:
    - model_used (constant): A constant representing the model to be used (RIDGE, KNN, or DT).
    - param (int): The parameter value to configure the model. Represents alpha for Ridge, 
      n_neighbors for KNN, and max_depth for DT.
    - rand_value (int): A multiplier for the random state, used in models that support it.

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


def plot_model_q2_4(model_used, complex_param, param, variable_size_LS, expect_error, var):
    """
    Plots and saves a graph comparing expected error and variance for a model over different sample sizes,
    and highlights the bias^2 + residual error.

    Parameters:
    - model_used (constant): A constant representing the model used (RIDGE, KNN, or DT).
    - complex_param (str): The name of the complexity parameter for the model.
    - param (int): The value of the complexity parameter.
    - variable_size_LS (list): List of sample sizes.
    - expect_error (list): List of expected error values corresponding to each sample size.
    - var (list): List of variance values corresponding to each sample size.

    Returns:
    - None
    """

    plt.figure()
    plt.plot(variable_size_LS, expect_error, label="Expected error")
    plt.plot(variable_size_LS, var, label="Variance")
    plt.plot(variable_size_LS, expect_error - var,
             label=r"$\mathrm{bias^2 + residual \ error}$")

    plt.legend(loc='best')
    plt.xlabel("Sample Size")
    plt.ylabel("Error")
    plt.title("Error "+str(model_used)+" related to learning sample size with " +
              str(complex_param)+"="+str(param))
    plt.savefig(
        f'2.4_{model_used}_{param}-{MIN_LS}-{MAX_LS}-{STEP_LS}.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    # list of model to test
    models = [RIDGE, KNN, DT]
    # list main complexity param
    complex_params = [r'($\mathrm{max\_depth}$)',
                      r'($\mathrm{k}$)', r'($\mathrm{\lambda^{3}}$)']

    # disgusting
    choice_model = RIDGE

    if choice_model == DT:
        model_used = DT
        hyper_params = DEPTHS_FIXED
        complex_param = complex_params[0]
    if choice_model == KNN:
        model_used = KNN
        hyper_params = K_FIXED
        complex_param = complex_params[1]
    if choice_model == RIDGE:
        model_used = RIDGE
        hyper_params = ALPHA_FIXED
        complex_param = complex_params[2]

    X, y = load_superconduct()

    # separation dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RAND_STATE)

    # nb different learning simple
    number_LS = 50

    # different size LS
    variable_size_LS = np.arange(MIN_LS, MAX_LS, STEP_LS)

    true_error = np.empty(number_LS)
    var = np.empty(len(variable_size_LS))
    expect_error = np.empty(len(variable_size_LS))
    predictions = np.empty((number_LS, X_test.shape[0]))

    for idx_param, param in enumerate(hyper_params):

        for index, ls_size in enumerate(variable_size_LS):

            model = model_choice(model_used, param, ls_size)

            # for on multiple LS to average
            for ls in range(number_LS):

                # shuffle training set and take the ls_size first to train
                X_t, y_t = shuffle(
                    X_train, y_train, random_state=(RAND_STATE * ls))
                model.fit(X_t[:ls_size], y_t[:ls_size])

                predictions[ls] = model.predict(X_test)
                true_error[ls] = mean_squared_error(y_test, predictions[ls])

            # var and except_error over all LS
            var[index] = np.mean(np.var(predictions, axis=0))
            expect_error[index] = np.mean(np.mean(true_error))

        plot_model_q2_4(model_used, complex_param, param,
                        variable_size_LS, expect_error, var)
