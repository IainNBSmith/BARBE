import numpy as np
from numpy.random import Generator, PCG64
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from barbe.discretizer import CategoricalEncoder
import matplotlib.pyplot as plt
import warnings


def simulate_simple_classified(seed_num=52146, size=1000, clusters_keep=1):
    matrix_size = 2
    sigma = np.array([[1.2+40,  0.03],  # add 5 to scale to see more
                      [0.03,   2+10]])
    mu = np.array([-1-3, 3])
    #sigma2 = np.array([[0.3,  1,  1.8, -2],
    #                   [1,    2,  1,    1],
    #                   [1.8,  1,  1,   -0.8],
    #                   [-2,   1, -0.8,  0.008]])
    #mu2 = np.array([-10, 0, -1, 1])
    sigma2 = np.array([[0.3,  0.8],
                      [0.8,    1.8+30]])
    mu2 = np.array([3+10, 1-3])

    # **************** TESTING PROTOCOL **************** #
    # In testing we aim to have a better performance when the models align during
    # lin_reg_params_i[-1] = bias
    lin_reg_params_i = [-2, 5, -2]
    # [i][j] i < j
    lin_reg_params_ij = [[None, 2.6]]

    def classifier_function(x):
        class_sum = lin_reg_params_i[-1]
        for i in range(matrix_size):
            class_sum += lin_reg_params_i[i] * x[i] if i == 1 else lin_reg_params_i[i]
            for j in range(i+1, matrix_size):
                class_sum += lin_reg_params_ij[i][j] * x[i] * x[j]

        return 1 if 30 > class_sum > -50 else 0

    random_state = np.random.default_rng(seed=seed_num)
    if clusters_keep == 1:
        X = random_state.multivariate_normal(mu, sigma, size=size)
        z = np.repeat(1, size)
    elif clusters_keep == 2:
        X = random_state.multivariate_normal(mu2, sigma2, size=size)
        z = np.repeat(2, size)
    else:
        X = random_state.multivariate_normal(mu, sigma, size=round((3 * size) / 4))
        X2 = random_state.multivariate_normal(mu2, sigma2, size=round((1 * size) / 4))
        X = np.vstack([X, X2])
        z = np.repeat(1, round((3 * size) / 4))
        z2 = np.repeat(2, round((1 * size) / 4))
        z = np.concatenate([z, z2])
    y = []
    for ii in range(X.shape[0]):
        y.append(classifier_function(X[ii, :]))
    y = np.array(y)

    return X, y, z


def simulate_linear_classified(seed_num=52146, size=1000, clusters_keep=1):
    matrix_size = 4
    sigma = np.array([[0.1,  0,  0.5, -0.3],
                      [0,    2,  0,    0],
                      [0.5,  0,  4,   -0.8],
                      [-0.3, 0, -0.8,  3]])
    mu = np.array([-1, 0, 1, 10])
    #sigma2 = np.array([[0.3,  1,  1.8, -2],
    #                   [1,    2,  1,    1],
    #                   [1.8,  1,  1,   -0.8],
    #                   [-2,   1, -0.8,  0.008]])
    #mu2 = np.array([-10, 0, -1, 1])
    sigma2 = np.array([[0.2, 0,   0.5, -0.3],
                      [0,    2.4, 0,    0],
                      [0.5,  0,   4.1, -0.8],
                      [-0.3, 0,  -0.8,  2.7]]) + 0.6
    mu2 = np.array([-3, 0.7, 3, 7.4])

    # **************** TESTING PROTOCOL **************** #
    # In testing we aim to have a better performance when the models align during
    # lin_reg_params_i[-1] = bias
    lin_reg_params_i = [-2, 10, 6, -0.8, 2]
    # [i][j] i < j
    lin_reg_params_ij = [[None, 0, 0, 2],
                         [None, None, -2.3, 0],
                         [None, None, None, 0.7],
                         [None]]
    # [i][j][k]
    lin_reg_params_ijk = [[[],
                           [None, None, 0.8, 0],     # 1, 2, ...
                           [None, None, None, 0],  # 1, 3, ...
                           []],
                          [[],
                           [],
                           [None, None, None, -1.7],  # 2, 3, ...
                           []],
                          [None],
                          [None]]

    def classifier_function(x):
        class_sum = lin_reg_params_i[-1]
        for i in range(matrix_size):
            class_sum += lin_reg_params_i[i] * x[i]
            for j in range(i+1, matrix_size):
                class_sum += lin_reg_params_ij[i][j] * x[i] * x[j]
                for k in range(j+1, matrix_size):
                    class_sum += lin_reg_params_ijk[i][j][k] * x[i] * x[j] * x[k]

        return 1 if class_sum > -15 else 0

    random_state = np.random.default_rng(seed=seed_num)
    if clusters_keep == 1:
        X = random_state.multivariate_normal(mu, sigma, size=size)
        z = np.repeat(1, size)
    elif clusters_keep == 2:
        X = random_state.multivariate_normal(mu2, sigma2, size=size)
        z = np.repeat(2, size)
    else:
        X = random_state.multivariate_normal(mu, sigma, size=round((3 * size) / 4))
        X2 = random_state.multivariate_normal(mu2, sigma2, size=round((1 * size) / 4))
        X = np.vstack([X, X2])
        z = np.repeat(1, round((2 * size) / 4))
        z2 = np.repeat(2, round((2 * size) / 4))
        z = np.concatenate([z, z2])
    y = []
    for ii in range(X.shape[0]):
        y.append(classifier_function(X[ii, :]))
    y = np.array(y)

    return X, y, z


if __name__ == '__main__':
    pass
