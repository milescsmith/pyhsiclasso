import importlib.resources as ir

import numpy as np
import pytest

from pyhsiclasso import HSICLasso


@pytest.fixture
def load_input_data():
    def _load_input_data(test_file):
        return ir.files("tests").joinpath("test_data", test_file)

    return _load_input_data


@pytest.fixture
def hsic_obj():
    return HSICLasso()


@pytest.fixture
def load_covars(hsic_obj, load_input_data, request):
    if request.param == [0, 0]:  # most of the time, we just want to return an empty array
        return np.array([])
    hsic_obj.input(load_input_data("matlab_data.mat"))
    return hsic_obj.X_in[request.param, :].T
