#!/usr/bin.env python

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy.testing as nptst
import pytest

from pyhsiclasso.api import HSICLasso

warnings.simplefilter("always")


@pytest.mark.parametrize(
    "input_data,num_feat,test_b,test_m,expected_a,load_covars",
    [
        ("matlab_data.mat", 5, 0, 3, [1099, 99, 199, 1299, 299], [0, 0]),
        ("matlab_data.mat", 10, 0, 3, [1099, 99, 199, 1299, 1477, 1405, 1073, 299, 1596, 358], [0, 0]),
        ("matlab_data.mat", 5, 50, 10, [1099, 99, 199, 299, 1299], [0, 0]),
        ("matlab_data.mat", 10, 50, 10, [1099, 99, 199, 1477, 299, 1299, 1073, 1405, 358, 1596], [0, 0]),
        ("matlab_data.mat", 5, 0, 3, [199, 1477, 1405, 1073, 1596], [99, 299]),
        pytest.param(None, None, None, None, None, [0, 0], marks=pytest.mark.xfail(raises=UnboundLocalError)),
    ],
    indirect=["load_covars"],
)
def test_regression(
    hsic_obj: HSICLasso,
    load_input_data: Callable[..., Any],
    input_data: Literal["matlab_data.mat"] | None,
    num_feat: int | None,
    test_b: int | None,
    test_m: int | None,
    expected_a: list[int] | None,
    load_covars: list[int],
):
    if input_data:
        hsic_obj.input(load_input_data(input_data))
    hsic_obj.regression(num_feat, B=test_b, M=test_m, covars=load_covars)
    assert hsic_obj.A == expected_a


@pytest.mark.parametrize("input_data", ["csv_data.csv"])
def test_regression_non_divisor_block_size(
    hsic_obj: HSICLasso, load_input_data: Callable[..., Any], input_data: Literal["csv_data.csv"]
):
    with warnings.catch_warnings(record=True) as w:
        hsic_obj.input(load_input_data(input_data))
        B = int(hsic_obj.X_in.shape[1] / 2) - 1
        n = hsic_obj.X_in.shape[1]
        numblocks = n / B
        hsic_obj.regression(10, B, 10)

        nptst.assert_equal(
            hsic_obj.A,
            [1422, 248, 512, 1581, 1670, 764, 1771, 896, 779, 398],
        )
        assert len(w) == 1
        assert w[-1].category is RuntimeWarning
        assert (
            str(w[-1].message)
            == f"B {B} must be an exact divisor of the number of samples {n}. Number of blocks {numblocks} will be approximated to {int(numblocks)}."
        )
