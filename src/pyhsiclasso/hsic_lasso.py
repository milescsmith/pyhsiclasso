#!/usr/bin/env python

from typing import Literal

import numpy as np
from joblib import Parallel, delayed

from pyhsiclasso.kernel_tools import kernel_delta_norm, kernel_gaussian, kernel_type


def hsic_lasso(
    X: np.array,
    Y: np.array,
    y_kernel: str,
    x_kernel: kernel_type = "Gaussian",
    n_jobs: int = -1,
    discarded: int = 0,
    B: int = 0,
    M: int = 1,
) -> tuple[np.array, np.array]:
    """
    Input:
        X      input_data
        Y      target_data
        y_kernel  We employ the Gaussian kernel for inputs. For output kernels,
                  we use the Gaussian kernel for regression cases and
                  the delta kernel for classification problems.
    Output:
        X         matrix of size d x (n * B (or n) * M)
        X_ty      vector of size d x 1
    """
    d, n = X.shape
    # dy = Y.shape[0]

    L = compute_kernel(Y, y_kernel, B, M, discarded)
    L = np.reshape(L, (n * B * M, 1))

    # Preparing design matrix for HSIC Lars
    result = Parallel(n_jobs=n_jobs)(
        [
            delayed(parallel_compute_kernel)(np.reshape(X[k, :], (1, n)), x_kernel, k, B, M, n, discarded)
            for k in range(d)
        ]
    )

    # non-parallel version for debugging purposes
    # result = []
    # for k in range(d):
    #     X = parallel_compute_kernel(X[k, :], x_kernel, k, B, M, n, discarded)
    #     result.append(X)

    result = dict(result)

    K = np.array([result[k] for k in range(d)]).T
    KtL = np.dot(K.T, L)

    return K, KtL, L


def compute_kernel(x: np.array, kernel: kernel_type, B: int = 0, M: int = 1, discarded: int = 0):
    d, n = x.shape

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32)

    # Normalize data
    if kernel == "Gaussian":
        x = (x / (x.std() + 10e-20)).astype(np.float32)

    st = 0
    ed = B**2
    index = np.arange(n)
    for m in range(M):
        np.random.seed(m)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            if kernel == "Gaussian":
                k = kernel_gaussian(x[:, index[i:j]], x[:, index[i:j]], np.sqrt(d))
            elif kernel == "Delta":
                k = kernel_delta_norm(x[:, index[i:j]], x[:, index[i:j]])

            k = np.dot(np.dot(H, k), H)

            # Normalize HSIC tr(k*k) = 1
            k = k / (np.linalg.norm(k, "fro") + 10e-10)
            K[st:ed] = k.flatten()
            st += B**2
            ed += B**2

    return K


def parallel_compute_kernel(x: np.array, kernel: kernel_type, feature_idx: int, B: int, M: int, n: int, discarded: int):
    return (feature_idx, compute_kernel(x, kernel, B, M, discarded))
