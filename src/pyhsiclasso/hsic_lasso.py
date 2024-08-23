#!/usr/bin/env python

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed, parallel_config

from pyhsiclasso.kernel_tools import kernel_delta_norm, kernel_gaussian


def hsic_lasso(
    x: npt.NDArray,
    y: npt.NDArray,
    y_kernel: Literal["Delta_norm", "Delta", "Gaussian"] = "Gaussian",
    x_kernel: Literal["Delta_norm", "Delta", "Gaussian"] = "Gaussian",
    n_jobs: int = -1,
    discarded: int = 0,
    b: int = 0,
    M: int = 1,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Input:
        x      input_data
        y      target_data
        y_kernel  We employ the Gaussian kernel for inputs. For output kernels,
                  we use the Gaussian kernel for regression cases and
                  the delta kernel for classification problems.
    Output:
        x         matrix of size d x (n * B (or n) * M)
        x_ty      vector of size d x 1
    """
    d, n = x.shape

    kernel_res = compute_kernel(y, y_kernel, b, M, discarded)
    kernel_res = np.reshape(kernel_res, (n * b * M, 1))

    # Preparing design matrix for HSIC Lars
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")
        with parallel_config(backend="loky", n_jobs=n_jobs):
            result = Parallel()(
                [delayed(parallel_compute_kernel)(np.reshape(x[k, :], (1, n)), x_kernel, k, b, M, discarded) for k in range(d)]
            )

    # non-parallel version for debugging purposes
    # result = [parallel_compute_kernel(np.reshape(X[k, :], (1, n)), x_kernel, k, B, M, discarded) for k in trange(d)]
    # for k in trange(d):
    #     result.append()

    result = dict(result)

    k = np.array([result[_] for _ in range(d)]).T
    ktl = np.dot(k.T, kernel_res)

    return k, ktl, kernel_res


def compute_kernel(
    x: npt.NDArray, kernel: Literal["Delta_norm", "Delta", "Gaussian"], b: int = 0, m: int = 1, discarded: int = 0
) -> npt.NDArray:
    d, n = x.shape

    h = np.eye(b, dtype=np.float32) - 1 / b * np.ones(b, dtype=np.float32)
    k = np.zeros(n * b * m, dtype=np.float32)

    # Normalize data
    if kernel == "Gaussian":
        x = (x / (x.std() + 10e-20)).astype(np.float32)

    st = 0
    ed = b**2
    index = np.arange(n)
    for _m in range(m):
        np.random.seed(_m)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, b):
            j = min(n, i + b)

            if kernel == "Gaussian":
                current_k = kernel_gaussian(x[:, index[i:j]], x[:, index[i:j]], np.sqrt(d))
            elif kernel == "Delta":
                current_k = kernel_delta_norm(x[:, index[i:j]], x[:, index[i:j]])

            current_k: npt.NDArray = np.dot(np.dot(h, current_k), h)

            # Normalize HSIC tr(k*k) = 1
            current_k = current_k / (np.linalg.norm(current_k, "fro") + 10e-10)
            k[st:ed] = current_k.flatten()
            st += b**2
            ed += b**2

    return k


def parallel_compute_kernel(
    x: np.array,
    kernel: Literal["Delta_norm", "Delta", "Gaussian"],
    feature_idx: int,
    B: int,
    M: int,
    discarded: int,
) -> tuple[int, npt.NDArray]:
    return (feature_idx, compute_kernel(x, kernel, B, M, discarded))
