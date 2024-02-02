#!/usr/bin.env python
# coding: utf-8
import importlib.resources as ir

import pytest

from pyhsiclasso import input_file


@pytest.fixture(params=["csv_data.csv", "tsv_data.tsv", "matlab_data.mat"])
def input_data(request):
    return ir.files("tests").joinpath("test_data", request.param)


@pytest.mark.parametrize("num_col", [62, 62, 100])
def test_input(input_data, num_col):
    X_in, Y_in, _ = input_file(input_data)
    X_in_row, X_in_col = X_in.shape
    Y_in_row, Y_in_col = Y_in.shape
    assert X_in_row == 2000
    assert X_in_col == 62
    assert Y_in_row == 1
    assert Y_in_col == 62