#!/usr/bin/env python
import numpy as np
import numpy.typing as npt


def kernel_delta_norm(x_in_1: npt.NDArray, x_in_2: npt.NDArray) -> npt.NDArray:
    n_1 = x_in_1.shape[1]
    n_2 = x_in_2.shape[1]
    k = np.zeros((n_1, n_2))
    u_list = np.unique(x_in_1)
    for ind in u_list:
        c_1 = np.sqrt(np.sum(x_in_1 == ind))
        c_2 = np.sqrt(np.sum(x_in_2 == ind))
        ind_1 = np.where(x_in_1 == ind)[1]
        ind_2 = np.where(x_in_2 == ind)[1]
        k[np.ix_(ind_1, ind_2)] = 1 / c_1 / c_2
    return k


def kernel_delta(x_in_1: npt.NDArray, x_in_2: npt.NDArray) -> npt.NDArray:
    n_1 = x_in_1.shape[1]
    n_2 = x_in_2.shape[1]
    k = np.zeros((n_1, n_2))
    u_list = np.unique(x_in_1)
    for ind in u_list:
        ind_1 = np.where(x_in_1 == ind)[1]
        ind_2 = np.where(x_in_2 == ind)[1]
        k[np.ix_(ind_1, ind_2)] = 1
    return k


def kernel_gaussian(x_in_1: npt.NDArray, x_in_2: npt.NDArray, sigma: np.float64) -> npt.NDArray:
    n_1 = x_in_1.shape[1]
    n_2 = x_in_2.shape[1]
    x_in_12 = np.sum(np.power(x_in_1, 2), 0)
    x_in_22 = np.sum(np.power(x_in_2, 2), 0)
    dist_2 = np.tile(x_in_22, (n_1, 1)) + np.tile(x_in_12, (n_2, 1)).transpose() - 2 * np.dot(x_in_1.T, x_in_2)
    return np.exp(-dist_2 / (2 * np.power(sigma, 2)))
