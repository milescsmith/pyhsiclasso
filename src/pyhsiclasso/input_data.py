#!/usr/bin/env python

from pathlib import Path

import numpy.typing as npt
import pandas as pd
from loguru import logger
from scipy import io as spio


def input_txt_file(
    file_name: Path | str, output: str | list[str], sep: str | None = None, featname: list[str] | None = None
) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
    df = pd.read_csv(file_name, sep=sep)

    return input_df(df=df, output=output, featname=featname)


def input_df(
    df: pd.DataFrame, output: str | list[str] | None = None, featname: list[str] | None = None
) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
    match output:
        case str():
            output = [output]
        case None:
            output = ["class"]
        case _:
            msg = "Dude, come on..."
            raise ValueError(msg)
    featname = df.drop(columns=output).columns.intersection(df.columns).to_list() if featname is None else featname

    x_in = df.loc[:, featname].values.T

    try:
        if len(output) == 1:
            y_in = df.loc[:, output].values.reshape(1, len(df.index))
        else:
            y_in = df.loc[:, output].values.T
    except KeyError as e:
        logger.exception(f"{e=}: {output} was not found as a column name")
        raise

    return x_in, y_in, featname


def input_matlab_file(file_name) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
    data = spio.loadmat(file_name)

    if "X" in data.keys() and "Y" in data.keys():
        x_in = data["X"]
        y_in = data["Y"]
    elif "X_in" in data.keys() and "Y_in" in data.keys():
        x_in = data["X_in"]
        y_in = data["Y_in"]
    elif "x" in data.keys() and "y" in data.keys():
        x_in = data["x"]
        y_in = data["y"]
    elif "x_in" in data.keys() and "y_in" in data.keys():
        x_in = data["x_in"]
        y_in = data["y_in"]
    else:
        msg = "not find input data"
        raise KeyError(msg)

    # Create feature list
    d = x_in.shape[0]
    featname = [("%d" % i) for i in range(1, d + 1)]

    return x_in, y_in, featname


def input_file(
    file_name: Path | str, output: str | list[str] = "class", **kwargs
) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
    file_name = Path(file_name) if isinstance(file_name, str) else file_name
    if not file_name.exists():
        msg = f"{file_name} was not found"
        raise FileNotFoundError(msg)

    match file_name.suffix:
        case ".csv":
            x_in, y_in, featname = input_txt_file(file_name, output=output, sep=",")
        case ".tsv":
            x_in, y_in, featname = input_txt_file(file_name, output=output, sep="\t")
        case ".mat":
            x_in, y_in, featname = input_matlab_file(file_name)
        case ".txt":
            x_in, y_in, featname = input_txt_file(file_name, **kwargs)
    return x_in, y_in, featname
