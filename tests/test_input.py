#!/usr/bin.env python

import importlib.resources as ir
from importlib.abc import Traversable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from pyhsiclasso import HSICLasso


def test_syntax(hsic_obj: HSICLasso):
    with pytest.raises(SyntaxError) as exc_info:
        hsic_obj._check_args([])
        assert exc_info.type is SyntaxError


@pytest.mark.parametrize("arg", ["txt", "hoge.txt", ("hogecsv")])
def test_value(hsic_obj: HSICLasso, arg: Literal["txt", "hoge.txt", "hogecsv"]):
    with pytest.raises(FileNotFoundError) as exc_info:
        hsic_obj._check_args([arg])
        assert exc_info.type is FileNotFoundError


@pytest.mark.parametrize(
    "arg",
    [
        (1, 2, 3),
        (123),
        ([1, 2, 3]),
        (np.array([1, 2, 3])),
        ("hoge", "hoge"),
        ("hoge", [1, 2, 3]),
        ([1, 2, 3], "hoge"),
        ("hoge", np.array([1, 2, 3])),
        (np.array([1, 2, 3]), "hoge"),
        (123, [1, 2, 3]),
        ([1, 2, 3], 123),
        (123, np.array([1, 2, 3])),
        (np.array([1, 2, 3]), 123),
        ([1, 2, 3], np.array([1, 2, 3])),
        (np.array([1, 2, 3]), [1, 2, 3]),
    ],
)
def test_type(hsic_obj: HSICLasso, arg: list[int] | npt.ArrayLike | int | str):
    with pytest.raises(TypeError) as exc_info:
        hsic_obj._check_args([arg])
        assert exc_info.type is TypeError


@pytest.fixture(params=["csv_data.csv", "tsv_data.tsv", "matlab_data.mat"])
def input_data(request: pytest.FixtureRequest):
    return ir.files("tests").joinpath("test_data", request.param)


def test_file_found(hsic_obj: HSICLasso, input_data: Traversable):
    assert hsic_obj._check_args([input_data])


def test_file_input(hsic_obj: HSICLasso, input_data: Traversable):
    assert hsic_obj.input(input_data)


@pytest.mark.parametrize("arg", [[np.array([1, 2, 3]), np.array([1, 2, 3])], [[1, 2, 3], [1, 2, 3]]])
def test_proper_args(hsic_obj: HSICLasso, arg: list[npt.ArrayLike] | list[list[int]]):
    assert hsic_obj._check_args(arg)


@pytest.mark.parametrize(
    "X_in,Y_in,expected_x_in_row,expected_x_in_col,expected_y_in_row,expected_y_in_col",
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
            marks=pytest.mark.xfail(raises=ValueError)
        ),
    ],
)
def test_input_data_list(
    hsic_obj: HSICLasso,
    X_in: list[list[int]],
    Y_in: list[int],
    expected_x_in_row: int,
    expected_x_in_col: int,
    expected_y_in_row: int,
    expected_y_in_col: int,
):
    hsic_obj._input_data_list(X_in, Y_in)
    assert (expected_x_in_row, expected_x_in_col) == hsic_obj.X_in.shape
    assert (expected_y_in_row, expected_y_in_col) == hsic_obj.Y_in.shape


@pytest.mark.parametrize(
    "X_in,Y_in,expected_x_in_row,expected_x_in_col,expected_y_in_row,expected_y_in_col",
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
            0,0,0,0,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
    ],
)
def test_input_data_ndarray(
    hsic_obj: HSICLasso,
    X_in: list[list[int]],
    Y_in: list[int],
    expected_x_in_row: int,
    expected_x_in_col: int,
    expected_y_in_row: int,
    expected_y_in_col: int,
):
    hsic_obj._input_data_list(X_in, Y_in)
    assert (expected_x_in_row, expected_x_in_col) == hsic_obj.X_in.shape
    assert (expected_y_in_row, expected_y_in_col) == hsic_obj.Y_in.shape

@pytest.mark.parametrize(
    "X_in,Y_in",
    [
        ([[1, 1, 1], [2, 2, 2]], [1, 2]),
        (
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
            [1, 2, 3, 4],
        ),
        ([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], [1, 2, 3, 4]),
        pytest.param([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]], [1, 2, 3, 4], marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_check_shape(hsic_obj: HSICLasso, X_in: list[int], Y_in: list[int]):
    hsic_obj._input_data_list(X_in, Y_in)
    assert hsic_obj._check_shape()
