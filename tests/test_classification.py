#!/usr/bin.env python

import warnings
from collections.abc import Callable
from typing import Any, Literal

import pytest

from pyhsiclasso.api import HSICLasso


@pytest.mark.parametrize(
    "input_data,num_feat,b_divisor,m,discrete_x,load_covars,expected_A",
    [
        pytest.param(None, None, None, None, False, [0, 0], None, marks=pytest.mark.xfail(raises=UnboundLocalError)),
        (
            "csv_data.csv",
            5,
            0,
            3,
            True,
            [0, 0],
            [764, 1422, 512, 248, 1581],
        ),
        (
            "csv_data.csv",
            10,
            0,
            3,
            True,
            [0, 0],
            [764, 1422, 512, 248, 1581, 1670, 1771, 896, 779, 266],
        ),
        (
            "csv_data.csv",
            5,
            2,
            10,
            True,
            [0, 0],
            [764, 1422, 512, 248, 266],
        ),
        (
            "csv_data.csv",
            10,
            2,
            10,
            True,
            [0, 0],
            [764, 1422, 512, 248, 1670, 1581, 266, 896, 1771, 779],
        ),
        (
            "matlab_data.mat",
            5,
            0,
            3,
            False,
            [1422, 512],
            [622, 841, 1636, 1891, 116],
        ),
    ],
    indirect=["load_covars"],
    scope="function",
)
def test_classification(
    hsic_obj: HSICLasso,
    load_input_data: Callable[..., Any],
    input_data: Literal["csv_data.csv", "matlab_data.mat"] | None,
    num_feat: int | None,
    b_divisor: int | None,
    m: int | None,
    discrete_x: bool,
    load_covars: list[int],
    expected_A: list[int] | None,
):
    if not input_data:
        hsic_obj.classification()

    hsic_obj.input(load_input_data(input_data))
    B = int(hsic_obj.X_in.shape[1] / b_divisor) if b_divisor > 0 else 0
    hsic_obj.classification(num_feat=num_feat, B=B, M=m, discrete_x=discrete_x, n_jobs=1, covars=load_covars)

    assert hsic_obj.A == expected_A


@pytest.mark.parametrize(
    "input_data,num_feat,b_divisor,m,load_covars,expected_A",
    [("csv_data.csv", 10, 2, 10, [0, 0], [1422, 764, 512, 248, 1670, 1581, 896, 266, 1771, 779])],
    indirect=["load_covars"],
)
def test_classification_non_divisor_block_size(
    hsic_obj: HSICLasso,
    load_input_data: Callable[..., Any],
    input_data: Literal["csv_data.csv"],
    num_feat: int,
    b_divisor: int,
    m: int,
    load_covars: list[int],
    expected_A: list[int],
):
    # use non-divisor as block size
    with warnings.catch_warnings(record=True) as w:
        hsic_obj.input(load_input_data(input_data))
        B = int(hsic_obj.X_in.shape[1] / b_divisor) - 1 if b_divisor > 0 else 0
        n = hsic_obj.X_in.shape[1]
        numblocks = n / B

        hsic_obj.classification(num_feat=num_feat, B=B, M=m, discrete_x=True, covars=load_covars)
        assert hsic_obj.A == expected_A
        assert len(w) == 1
        assert w[-1].category == RuntimeWarning
        assert (
            str(w[-1].message)
            == f"B {B} must be an exact divisor of the number of samples {n}. Number of blocks {numblocks} will be approximated to {int(numblocks)}."
        )
