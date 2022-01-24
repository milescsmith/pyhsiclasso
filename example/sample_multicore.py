#!/usr/bin.env python
# coding: utf-8
from pyhsiclasso import HSICLasso


def main():
    hsic_lasso = HSICLasso()
    hsic_lasso.input("../tests/test_data/matlab_data.mat")

    # Single core processing
    hsic_lasso.regression(5, n_jobs=1)

    # Multi-core processing. Use all available cores (default)
    hsic_lasso.regression(5, n_jobs=-1)


if __name__ == "__main__":
    main()
