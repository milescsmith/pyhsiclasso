#!/usr/bin.env python
import importlib.resources as ir

from pyhsiclasso import HSICLasso


def main():
    hsic_lasso = HSICLasso()
    hsic_lasso.input(ir.files("tests").joinpath("test_data", "csv_data_mv.csv"), output=["output1", "output2"]
    )
    hsic_lasso.regression(5)
    hsic_lasso.dump()
    hsic_lasso.plot_path()


if __name__ == "__main__":
    main()
