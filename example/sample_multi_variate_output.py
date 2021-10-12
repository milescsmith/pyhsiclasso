#!/usr/bin.env python
# coding: utf-8

from pyhsiclasso import HSICLasso



def main():
    hsic_lasso = HSICLasso()
    hsic_lasso.input(
        "../tests/test_data/csv_data_mv.csv", output_list=["output1", "output2"]
    )
    hsic_lasso.regression(5)
    hsic_lasso.dump()
    hsic_lasso.plot_path()


if __name__ == "__main__":
    main()
