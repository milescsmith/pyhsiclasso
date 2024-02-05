#!/usr/bin/env python

from pathlib import Path
import pandas as pd
from scipy import io as spio


def input_csv_file(file_name, output_list: list[str] | None = None):
    if output_list is None:
        output_list = ["class"]
    return input_txt_file(file_name, output_list, ",")


def input_tsv_file(file_name, output_list=None):
    if output_list is None:
        output_list = ["class"]
    return input_txt_file(file_name, output_list, "\t")


def input_txt_file(file_name: str, output_list: list[str], sep: str):
    df = pd.read_csv(file_name, sep=sep)

    # Store the column name (Feature name)
    featname = df.columns.tolist()
    input_index = list(range(len(featname)))
    output_index = []

    for output_name in output_list:
        if output_name not in featname:
            msg = f"Output variable, {output_name}, not found"
            raise ValueError(msg)

        tmp = featname.index(output_name)
        output_index.append(tmp)
        input_index.remove(tmp)
    for output_name in output_list:
        featname.remove(output_name)

    X_in = df.iloc[:, input_index].values.T

    if len(output_index) == 1:
        Y_in = df.iloc[:, output_index].values.reshape(1, len(df.index))
    else:
        Y_in = df.iloc[:, output_index].values.T

    return X_in, Y_in, featname


def input_df(df: pd.DataFrame, output_list: list[str] | None = None, featname: list[str] | None = None):
    if output_list is None:
        output_list = ["class"]
    # Store the column name (Feature name)
    if featname is None:
        featname = list(df.columns.drop(output_list))

    output_name = [_ for _ in output_list if _ in df.columns]

    X_in = df.loc[:, featname].values.T

    if len(output_name) == 1:
        Y_in = df.loc[:, output_name].values.reshape(1, len(df.index))
    elif len(output_name) > 1:
        Y_in = df.loc[:, output_name].values.T
    else:
        msg = f"Output variable, {output_name}, not found"
        raise ValueError(msg)
    print(f"X_in: {X_in.shape}, Y_in: {Y_in.shape}")
    return X_in, Y_in, featname


def input_matlab_file(file_name):
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


def input_file(file_name: Path | str, **kwargs) -> tuple:
    file_name = Path(file_name) if isinstance(file_name, str) else file_name
    print(f"file is {file_name} of type {type(file_name)}")
    match file_name.suffix:
        case ".csv":
            X_in, Y_in, featname = input_csv_file(file_name, **kwargs)
        case ".tsv":
            X_in, Y_in, featname = input_tsv_file(file_name, **kwargs)
        case ".mat":
            X_in, Y_in, featname = input_matlab_file(file_name)
        case ".txt":
            X_in, Y_in, featname = input_txt_file(file_name, **kwargs)
    return X_in, Y_in, featname
