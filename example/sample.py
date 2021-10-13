#!/usr/bin.env python
# coding: utf-8

import scipy.io as sio


from pyhsiclasso import HSICLasso


def main():

    # Numpy array input example
    hsic_lasso = HSICLasso()
    data = sio.loadmat("../tests/test_data/matlab_data.mat")
    X = data["X"].transpose()
    Y = data["Y"][0]
    featname = ["Feat%d" % x for x in range(1, X.shape[1] + 1)]

    hsic_lasso.input(X, Y, featname=featname)
    hsic_lasso.regression(5)
    hsic_lasso.dump()
    hsic_lasso.plot_path()

    # Save parameters
    hsic_lasso.save_param()


if __name__ == "__main__":
    main()
