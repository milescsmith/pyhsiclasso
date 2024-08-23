#!/usr/bin/env python
from typing import Final

import numpy as np
from rich import print as pp
from scipy.sparse import lil_matrix

A_VERY_SMALL_NUMBER: Final[int] = 1e-9


def nlars(X, x_ty, num_feat, max_neighbors):
    """
    We used the a Python implementation of the Nonnegative LARS solver
    written in MATLAB at http://orbit.dtu.dk/files/5618980/imm5523.zip

    Solves the problem argmin_beta 1/2||y-X*beta||_2^2  s.t. beta>=0.
    The problem is solved using a modification of the Least Angle Regression
    and Selection algorithm.
    As such the entire regularization path for the LASSO problem
    min 1/2||y-X*beta||_2^2 + lambda|beta|_1  s.t. beta>=0
    for all values of lambda is given in path.

    Input:
        X            matrix of size D x D
        X_ty         vector of size D x 1
        num_feat     the number of features you want to extract
    Output:
        path         the entire solution path
        beta         D x 1 solution vector
        A            selected features
        A_neighbors  related features of the selected features in A
        lam(lambda)  regularization value at beginning of step corresponds
                     to value of negative gradient
    """
    n, d = X.shape

    a_neighbors = []
    a_neighbors_score = []
    beta = np.zeros((d, 1), dtype=np.float32)
    path = lil_matrix((d, 4 * d))
    lam = np.zeros((1, 4 * d))

    indices = list(range(d))

    xtxbeta = np.dot(X.transpose(), np.dot(X, beta))
    c = x_ty - xtxbeta
    j = c.argmax()
    big_c = c[j]
    a = [indices[j]]
    indices.remove(indices[j])

    if len(big_c) == 0:
        lam[0] = 0
    else:
        lam[0, 0] = big_c[0]

    k = 0

    while sum(c[a]) / len(a) >= A_VERY_SMALL_NUMBER and len(a) < num_feat + 1:
        pp(f"{k=}")
        s = np.ones((len(a), 1), dtype=np.float32)

        try:
            w = np.linalg.solve(np.dot(X[:, a].transpose(), X[:, a]), s)
        except np.linalg.linalg.LinAlgError:
            # matrix is singular
            x_noisy = X[:, a] + np.random.normal(0, 10e-10, X[:, a].shape)
            w = np.linalg.solve(np.dot(x_noisy.transpose(), x_noisy), s)

        xtxw = np.dot(X.transpose(), np.dot(X[:, a], w))

        gamma1 = (big_c - c[indices]) / (xtxw[a[0]] - xtxw[indices])
        gamma2 = -beta[a] / (w)
        gamma3 = np.zeros((1, 1))
        gamma3[0] = c[a[0]] / (xtxw[a[0]])
        gamma = np.concatenate((np.concatenate((gamma1, gamma2)), gamma3))

        gamma[gamma <= A_VERY_SMALL_NUMBER] = np.inf
        t = gamma.argmin()
        mu = min(gamma)

        beta[a] = beta[a] + mu * w

        if t >= len(gamma1) and t < (len(gamma1) + len(gamma2)):
            lasso_cond = 1
            j = t - len(gamma1)
            indices.append(a[j])
            a.remove(a[j])
        else:
            lasso_cond = 0

        xtxbeta = np.dot(X.transpose(), np.dot(X, beta))
        c = x_ty - xtxbeta
        j = np.argmax(c[indices])
        big_c = max(c[indices])

        k += 1
        path[:, k] = beta

        if len(big_c) == 0:
            lam[k] = 0
        else:
            lam[0, k] = big_c[0]
        if lasso_cond == 0:
            a.append(indices[j])
            indices.remove(indices[j])

    # We run numfeat + 1 iteration to update beta and path information
    # Then, we return only numfeat features
    if len(a) > num_feat:
        a.pop()

    # Sort A with respect to beta
    s = beta[a]
    sort_index = sorted(range(len(s)), key=lambda k: s[k], reverse=True)

    a_sorted = [a[i] for i in sort_index]

    # Find nighbors of selected features
    xtxa = np.dot(X.transpose(), X[:, a_sorted])

    # Search up to 10 nighbors
    num_neighbors = max_neighbors + 1
    for i in range(len(a_sorted)):
        tmp = xtxa[:, i]
        sort_index = sorted(range(len(tmp)), key=lambda k: tmp[k], reverse=True)
        a_neighbors.append(sort_index[:num_neighbors])
        a_neighbors_score.append(tmp[sort_index[:num_neighbors]])

    path_final = path[:, 0 : (k + 1)].toarray()
    lam_final = lam[: k + 1]

    return path_final, beta, a_sorted, lam_final, a_neighbors, a_neighbors_score
