#!/usr/bin/env python

from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from pyhsiclasso.api import HSICLasso
from pyhsiclasso.hsic_lasso import hsic_lasso
from pyhsiclasso.input_data import input_file

try:
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

logger.disable(__package__)

__all__ = [
    "HSICLasso",
    "hsic_lasso",
    "input_file",
]
