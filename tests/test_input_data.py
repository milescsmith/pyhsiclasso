#!/usr/bin.env python
import pytest

from pyhsiclasso import input_file


@pytest.mark.parametrize(
    "input_data,x_shape,y_shape",
    [
        ("csv_data.csv", (2000, 62), (1, 62)),
        ("tsv_data.tsv", (2000, 62), (1, 62)),
        ("matlab_data.mat", (2000, 100), (1, 100)),
    ],
)
def test_input(load_input_data, input_data, x_shape, y_shape):
    X_in, Y_in, _ = input_file(load_input_data(input_data))
    assert x_shape == X_in.shape
    assert y_shape == Y_in.shape
