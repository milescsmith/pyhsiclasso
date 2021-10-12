#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from scipy import io as spio


def input_csv_file(file_name, output_list=["class"]):
    return input_txt_file(file_name, output_list, ",")


def input_tsv_file(file_name, output_list=["class"]):
    return input_txt_file(file_name, output_list, "\t")


def input_txt_file(file_name, output_list, sep):
    df = pd.read_csv(file_name, sep=sep)

    # Store the column name (Feature name)
    featname = df.columns.tolist()
    input_index = list(range(0, len(featname)))
    output_index = []

    for output_name in output_list:
        if output_name in featname:
            tmp = featname.index(output_name)
            output_index.append(tmp)
            input_index.remove(tmp)
        else:
            raise ValueError("Output variable, %s, not found" % (output_name))

    for output_name in output_list:
        featname.remove(output_name)

    X_in = df.iloc[:, input_index].values.T

    if len(output_index) == 1:
        Y_in = df.iloc[:, output_index].values.reshape(1, len(df.index))
    else:
        Y_in = df.iloc[:, output_index].values.T

    return X_in, Y_in, featname


def input_df(df: pd.DataFrame, output_list: list[str]=["class"], featname: list[str]=None):

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
        raise ValueError(f"Output variable, {output_name}, not found")
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
        raise KeyError("not find input data")

    # Create feature list
    d = X_in.shape[0]
    featname = [("%d" % i) for i in range(1, d + 1)]

    return X_in, Y_in, featname
