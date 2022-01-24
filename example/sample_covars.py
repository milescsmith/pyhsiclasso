#!/usr/bin.env python
# coding: utf-8


import numpy as np
import scipy.io as sio


from pyhsiclasso import HSICLasso


def main():

    # Numpy array input example
    hsic_lasso = HSICLasso()
    data = sio.loadmat("../tests/test_data/matlab_data.mat")
    X = data["X"].transpose()
    Y = data["Y"][0]
    featname = ["Feat%d" % x for x in range(1, X.shape[1] + 1)]

    # Get rid of the effect of feature 100 and 300
    covars_index = np.array([99, 299])

    hsic_lasso.input(X, Y, featname=featname)
    hsic_lasso.regression(5, covars=X[:, covars_index], covars_kernel="Gaussian")
    hsic_lasso.dump()
    hsic_lasso.plot_path()

    # Save parameters
    hsic_lasso.save_param()


if __name__ == "__main__":
    main()
