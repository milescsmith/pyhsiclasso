#!/usr/bin/env python

from pathlib import Path

import numpy.typing as npt
import pandas as pd
from loguru import logger
from scipy import io as spio


def input_txt_file(
    file_name: str, output: str | list[str], sep: str | None = None, featname: list[str] | None = None
) -> tuple[npt.ArrayLike, npt.ArrayLike, list[str]]:
    df = pd.read_csv(file_name, sep=sep)

    return input_df(df=df, output=output, featname=featname)


def input_df(
    df: pd.DataFrame, output: str | list[str] | None = None, featname: list[str] | None = None
) -> tuple[npt.ArrayLike, npt.ArrayLike, list[str]]:
    match output:
        case str():
            output = [output]
        case None:
            output = ["class"]
        case _:
            msg = "Dude, come on..."
            raise ValueError(msg)
    featname = df.drop(columns=output).columns.intersection(df.columns).to_list() if featname is None else featname

    X_in = df.loc[:, featname].values.T

    try:
        if len(output) == 1:
            Y_in = df.loc[:, output].values.reshape(1, len(df.index))
        else:
            Y_in = df.loc[:, output].values.T
    except KeyError as e:
        logger.exception(f"{e=}: {output} was not found as a column name")
        raise

    return X_in, Y_in, featname


def input_matlab_file(file_name) -> tuple[npt.ArrayLike, npt.ArrayLike, list[str]]:
    data = spio.loadmat(file_name)

    if "X" in data.keys() and "Y" in data.keys():
        X_in = data["X"]
        Y_in = data["Y"]
    elif "X_in" in data.keys() and "Y_in" in data.keys():
        X_in = data["X_in"]
        Y_in = data["Y_in"]
    elif "x" in data.keys() and "y" in data.keys():
        X_in = data["x"]
        Y_in = data["y"]
    elif "x_in" in data.keys() and "y_in" in data.keys():
        X_in = data["x_in"]
        Y_in = data["y_in"]
    else:
        msg = "not find input data"
        raise KeyError(msg)

    # Create feature list
    d = X_in.shape[0]
    featname = [("%d" % i) for i in range(1, d + 1)]

    return X_in, Y_in, featname


def input_file(
    file_name: Path | str, output: str | list[str] = "class", **kwargs
) -> tuple[npt.ArrayLike, npt.ArrayLike, list[str]]:
    file_name = Path(file_name) if isinstance(file_name, str) else file_name
    if not file_name.exists():
        msg = f"{file_name} was not found"
        raise FileNotFoundError(msg)

    match file_name.suffix:
        case ".csv":
            X_in, Y_in, featname = input_txt_file(file_name, output=output, sep=",")
        case ".tsv":
            X_in, Y_in, featname = input_txt_file(file_name, output=output, sep="\t")
        case ".mat":
            X_in, Y_in, featname = input_matlab_file(file_name)
        case ".txt":
            X_in, Y_in, featname = input_txt_file(file_name, **kwargs)
    return X_in, Y_in, featname
