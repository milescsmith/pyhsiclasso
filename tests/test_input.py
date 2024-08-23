#!/usr/bin.env python

import importlib.resources as ir
from importlib.resources.abc import Traversable
from contextlib import nullcontext

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pyhsiclasso import HSICLasso


@pytest.mark.parametrize(
    "input_data,output,featname",
    [
        pytest.param(1, 2, 3, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param([1, 2, 3], None, None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(123, None, None, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param("hoge", None, None, marks=pytest.mark.xfail(raises=FileNotFoundError)),
        pytest.param(np.array([1, 2, 3]), None, None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(
            pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD")),
            2,
            3,
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        pytest.param(
            pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD")),
            "E",
            3,
            marks=pytest.mark.xfail(raises=KeyError),
        ),
    ],
)
def test_bad_input(
    hsic_obj: HSICLasso,
    input_data: list[int] | npt.NDArray | int | str,
    output: list[int] | npt.NDArray | int | str,
    featname: list[int] | npt.NDArray | int | str,
):
    # if input_data is None:
    #     input_data =
    hsic_obj.input(input_data, output, featname)


@pytest.fixture(params=["csv_data.csv", "tsv_data.tsv", "matlab_data.mat"])
def input_data(request: pytest.FixtureRequest):
    return ir.files("tests").joinpath("data", request.param)


def test_file_input(hsic_obj: HSICLasso, input_data: Traversable):
    with nullcontext():
        hsic_obj.input(input_data)


@pytest.mark.parametrize(
    "x_in,y_in,expected_x_in_row,expected_x_in_col,expected_y_in_row,expected_y_in_col",
    [
        ([[1, 1, 1], [2, 2, 2]], [1, 2], 3, 2, 1, 2),
        ([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]], [1, 2, 3, 4], 5, 4, 1, 4),
        pytest.param(
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
            [[1, 2, 3, 4], [1, 2, 3, 4]],
            0,
            0,
            0,
            0,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_input_data_list(
    hsic_obj: HSICLasso,
    x_in: list[list[int]],
    y_in: list[int],
    expected_x_in_row: int,
    expected_x_in_col: int,
    expected_y_in_row: int,
    expected_y_in_col: int,
):
    hsic_obj._input_data_list(x_in, y_in)
    assert (expected_x_in_row, expected_x_in_col) == hsic_obj.x_in.shape
    assert (expected_y_in_row, expected_y_in_col) == hsic_obj.y_in.shape


@pytest.mark.parametrize(
    "x_in,y_in,expected_x_in_row,expected_x_in_col,expected_y_in_row,expected_y_in_col",
    [
        (np.array([[1, 1, 1], [2, 2, 2]]), np.array([1, 2]), 3, 2, 1, 2),
        (
            np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]),
            np.array([1, 2, 3, 4]),
            5,
            4,
            1,
            4,
        ),
        pytest.param(
            np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]),
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
            0,
            0,
            0,
            0,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_input_data_ndarray(
    hsic_obj: HSICLasso,
    x_in: list[list[int]],
    y_in: list[int],
    expected_x_in_row: int,
    expected_x_in_col: int,
    expected_y_in_row: int,
    expected_y_in_col: int,
):
    hsic_obj._input_data_list(x_in, y_in)
    assert (expected_x_in_row, expected_x_in_col) == hsic_obj.x_in.shape
    assert (expected_y_in_row, expected_y_in_col) == hsic_obj.y_in.shape


@pytest.mark.parametrize(
    "x_in,y_in",
    [
        ([[1, 1, 1], [2, 2, 2]], [1, 2]),
        (
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
            [1, 2, 3, 4],
        ),
        ([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], [1, 2, 3, 4]),
        pytest.param(
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
            [1, 2, 3, 4],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_check_shape(hsic_obj: HSICLasso, x_in: list[int], y_in: list[int]):
    hsic_obj._input_data_list(x_in, y_in)
    assert hsic_obj._check_shape()
