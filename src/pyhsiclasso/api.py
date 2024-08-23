#!/usr/bin/env python
import warnings
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from rich import print as pp
from rich.table import Table
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance

from pyhsiclasso.hsic_lasso import compute_kernel, hsic_lasso
from pyhsiclasso.input_data import input_file
from pyhsiclasso.nlars import nlars
from pyhsiclasso.plot_figure import plot_dendrogram, plot_heatmap, plot_path

TWO_DIMENSIONAL: Final[int] = 2


class HSICLasso:
    def __init__(self):
        self.input_file = None
        self.x_in = None
        self.y_in = None
        self.x = None
        self.xty = None
        self.path = None
        self.beta = None
        self.a = None
        self.a_neighbors = None
        self.a_neighbors_score = None
        self.lam = None
        self.featname = None
        self.linkage_dist = None
        self.hclust_featname = None
        self.hclust_featnameindex = None
        self.max_neighbors = 10

    def input(
        self,
        input_data: str | Path | npt.NDArray | pd.DataFrame | None = None,
        output: str | list[str] | npt.NDArray | pd.Series = "class",
        featname: list[str] | npt.NDArray | pd.Series | None = None,
    ):
        match input_data:
            case str() | Path():
                self._input_data_file(input_data, output)
            case np.ndarray() if not isinstance(output, list | np.ndarray | pd.Series):
                msg = (
                    "output is an invalid type. When input_list is a numpy array, "
                    "output must be a list, array, or Series with a length equal to the "
                    "number or rows in input_data"
                )
                raise ValueError(msg)
            case np.ndarray() if isinstance(output, list):
                self._input_data_ndarray(x_in=input_data, y_in=np.array(output), featname=featname)
            case np.ndarray():
                self._input_data_ndarray(x_in=input_data, y_in=output, featname=featname)
            case pd.DataFrame():
                self._input_data_dataframe(df=input_data, output=output, featname=featname)
            case _:
                input_data_type = str(type(input_data)).replace("<class ", "").rstrip(">").replace("'", "")
                msg = (
                    f"{input_data} is a {input_data_type}, "
                    f"input only knows how to deal with a str, Path, numpy array, or pandas dataframe"
                )
                raise TypeError(msg)

        if self.x_in is None or self.y_in is None:
            msg = "Check your input data"
            raise ValueError(msg)
        self._check_shape()
        logger.debug(f"Loaded a {self.x_in.shape[0]} x {self.x_in.shape[1]} matrix of data")

    def regression(
        self,
        num_feat: int = 5,
        b: int = 20,
        m: int = 3,
        discrete_x: bool = False,
        max_neighbors: int = 10,
        n_jobs: int = -1,
        covars: list[int] | npt.NDArray | None = None,
        covars_kernel: str = "Gaussian",
    ) -> bool:
        covars = np.array([]) if covars is None else covars

        self._run_hsic_lasso(
            num_feat=num_feat,
            y_kernel="Gaussian",
            b=b,
            m=m,
            discrete_x=discrete_x,
            max_neighbors=max_neighbors,
            n_jobs=n_jobs,
            covars=covars,
            covars_kernel=covars_kernel,
        )

        return True

    # TODO: okay, so covars really doesn't seem to work. categorical covars don't work. it only takes a numpy array, completely unlike the bare-ass readme
    def classification(
        self,
        num_feat: int = 5,
        b: int = 20,
        m: int = 3,
        discrete_x: bool = False,
        max_neighbors: int = 10,
        n_jobs: int = -1,
        covars: npt.NDArray | None = None,
        covars_kernel: Literal["Delta_norm", "Delta", "Gaussian"] = "Gaussian",
    ) -> bool:
        covars = np.array([]) if covars is None or not isinstance(covars, np.ndarray) else covars
        self._run_hsic_lasso(
            num_feat=num_feat,
            y_kernel="Delta",
            b=b,
            m=m,
            discrete_x=discrete_x,
            max_neighbors=max_neighbors,
            n_jobs=n_jobs,
            covars=covars,
            covars_kernel=covars_kernel,
        )

        return True

    def _run_hsic_lasso(
        self,
        y_kernel: Literal["Delta_norm", "Delta", "Gaussian"],
        num_feat: int,
        b: int,
        m: int,
        discrete_x: bool,
        max_neighbors: int,
        n_jobs: int,
        covars: npt.NDArray,  # TODO: Okay, but should this *really* be an array at this point?
        covars_kernel: str,
    ) -> bool:
        if self.x_in is None or self.y_in is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)
        self.max_neighbors = max_neighbors
        n = self.x_in.shape[1]
        b = b or n
        x_kernel: Literal["Delta_norm", "Delta", "Gaussian"] = "Delta" if discrete_x else "Gaussian"
        numblocks = n / b
        discarded = n % b
        pp(f"Block HSIC Lasso B = {b}.")

        if discarded:
            msg = (
                f"B {b} must be an exact divisor of the number of samples {n}. Number "
                f"of blocks {numblocks} will be approximated to {int(numblocks)}."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            numblocks = int(numblocks)

        # Number of permutations of the block HSIC
        m = 1 + bool(numblocks - 1) * (m - 1)
        pp(f"M set to {m}.")
        additional_text = " and Gaussian kernel for the covariates" if covars.size else ""
        pp(f"Using {x_kernel} kernel for the features, {y_kernel}kernel for the outcomes{additional_text}.")

        x, xty, ky = hsic_lasso(
            x=self.x_in,
            y=self.y_in,
            y_kernel=y_kernel,
            x_kernel=x_kernel,
            n_jobs=n_jobs,
            discarded=discarded,
            b=b,
            M=m,
        )

        # np.concatenate(self.X, axis = 0) * np.sqrt(1/(numblocks * M))
        self.x = x * np.sqrt(1 / (numblocks * m))
        self.xty = xty * 1 / (numblocks * m)

        if covars.size:
            if self.x_in.shape[1] != covars.shape[0]:
                msg = f"The number of rows in the covars matrix should be {self.x_in.shape[1]!s}"
                raise UnboundLocalError(msg)

            if covars_kernel == "Gaussian":
                kc = compute_kernel(covars.transpose(), "Gaussian", b, m, discarded)
            else:
                kc = compute_kernel(covars.transpose(), "Delta", b, m, discarded)
            kc = np.reshape(kc, (n * b * m, 1))

            ky = ky * np.sqrt(1 / (numblocks * m))
            kc = kc * np.sqrt(1 / (numblocks * m))

            betas = np.dot(ky.transpose(), kc) / np.trace(np.dot(kc.T, kc))
            # pp(betas)
            self.xty = self.xty - betas * np.dot(self.x.transpose(), kc)

        pp("Calculating nlars")
        (
            self.path,
            self.beta,
            self.a,
            self.lam,
            self.a_neighbors,
            self.a_neighbors_score,
        ) = nlars(self.x, self.xty, num_feat, self.max_neighbors)

        return True

    # For kernel Hierarchical Clustering
    def linkage(self, method="ward"):
        if self.a is None:
            msg = "Run regression/classification first"
            raise UnboundLocalError(msg)
        # selected feature name
        featname_index = []
        featname_selected = []
        for i in range(len(self.a) - 1):
            for index in self.a_neighbors[i]:
                if index not in featname_index:
                    featname_index.append(index)
                    featname_selected.append(self.featname[index])
        self.hclust_featname = featname_selected
        self.hclust_featnameindex = featname_index
        sim = np.dot(self.x[:, featname_index].transpose(), self.x[:, featname_index])
        dist = 1 - sim
        dist = np.maximum(0, dist - np.diag(np.diag(dist)))
        dist_sym = (dist + dist.transpose()) / 2.0
        self.linkage_dist = linkage(distance.squareform(dist_sym), method)

        return True

    def dump(self, num_neighbors: int = 5) -> Table:
        table_data = self.dump_dict()

        table = Table(title="HSIC-lasso results")
        table.add_column("Order", max_width=5, justify="right", no_wrap=True)
        table.add_column("Feature", justify="left", no_wrap=True)
        table.add_column("Score", justify="center", no_wrap=True)
        for i in range(num_neighbors):
            table.add_column("Related\nfeature\n" + f"{i+1}", justify="left", no_wrap=False, min_width=7)
            table.add_column("Related\nfeature\n" + f"{i+1} score", justify="center", no_wrap=False, min_width=10)

        for i, j in enumerate(table_data):
            new_list: list[str] = []
            for k in table_data[j]["related"]:
                new_list.extend((k, f'{table_data[j]["related"][k]:.3f}'))
            table.add_row(str(i), str(j), f'{table_data[j]["score"]:.3f}', *new_list)
        return table

    def dump_dict(self, num_heighbors: int = 5) -> dict:
        return {
            self.featname[self.a[i]]: {
                "order": i + 1,
                "score": self.beta[self.a[i]][0] / self.beta[self.a[0]][0],
                "related": {
                    self.featname[nn]: ns
                    for nn, ns, _ in zip(
                        self.a_neighbors[i][1:], self.a_neighbors_score[i][1:], range(num_heighbors), strict=False
                    )
                },
            }
            for i in range(len(self.a))
        }

    def plot_heatmap(self, filepath="heatmap.png"):
        if self.linkage_dist is None or self.hclust_featname is None or self.hclust_featnameindex is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)
        plot_heatmap(
            self.x_in[self.hclust_featnameindex, :],
            self.linkage_dist,
            self.hclust_featname,
            filepath,
        )
        return True

    def plot_dendrogram(self, filepath="dendrogram.png"):
        if self.linkage_dist is None or self.hclust_featname is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)
        plot_dendrogram(self.linkage_dist, self.hclust_featname, filepath)
        return True

    def plot_path(self, filepath="path.png"):
        if self.path is None or self.beta is None or self.a is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)
        plot_path(self.path, self.a, filepath)
        return True

    def get_features(self):
        index = self.get_index()

        return [self.featname[i] for i in index]

    def get_features_neighbors(self, feat_index=0, num_neighbors=5):
        index = self.get_index_neighbors(feat_index=feat_index, num_neighbors=num_neighbors)

        return [self.featname[i] for i in index]

    def get_index(self):
        return self.a

    def get_index_score(self):
        return self.beta[self.a, -1]

    def get_index_neighbors(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.a) - 1:
            msg = "Index does not exist"
            raise IndexError(msg)

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.a_neighbors[feat_index][1 : (num_neighbors + 1)]

    def get_index_neighbors_score(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.a) - 1:
            msg = "Index does not exist"
            raise IndexError(msg)

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.a_neighbors_score[feat_index][1 : (num_neighbors + 1)]

    def save_HSICmatrix(self, filename="HSICmatrix.csv"):
        if self.x_in is None or self.y_in is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)

        self.x, self.X_ty = hsic_lasso(self.x_in, self.y_in, "Gaussian")

        k = np.dot(self.x.transpose(), self.x)

        np.savetxt(filename, k, delimiter=",", fmt="%.7f")

        return True

    def save_score(self, filename="aggregated_score.csv"):
        maxval = self.beta[self.a[0]][0]

        with open(filename, "w") as fout:
            featscore = {}
            featcorrcoeff = {}
            for i in range(len(self.a)):
                hsic_xy = self.beta[self.a[i]][0] / maxval

                if self.featname[self.a[i]] not in featscore:
                    featscore[self.featname[self.a[i]]] = hsic_xy
                    corrcoeff = np.corrcoef(self.x_in[self.a[i]], self.y_in)[0][1]
                    featcorrcoeff[self.featname[self.a[i]]] = corrcoeff
                else:
                    featscore[self.featname[self.a[i]]] += hsic_xy

                for j in range(1, self.max_neighbors + 1):
                    hsic_xx = self.a_neighbors_score[i][j]
                    if self.featname[self.a_neighbors[i][j]] not in featscore:
                        featscore[self.featname[self.a_neighbors[i][j]]] = hsic_xy * hsic_xx
                        corrcoeff = np.corrcoef(self.x_in[self.a_neighbors[i][j]], self.y_in)[0][1]
                        featcorrcoeff[self.featname[self.a_neighbors[i][j]]] = corrcoeff
                    else:
                        featscore[self.featname[self.a_neighbors[i][j]]] += hsic_xy * hsic_xx

            # Sorting decending order
            featscore_sorted = sorted(featscore.items(), key=lambda i: i[1], reverse=True)

            # Add Pearson correlation for comparison
            fout.write("Feature,Score,Pearson Corr\n")
            for key, val in featscore_sorted:
                fout.write(f"{key},{val!s},{featcorrcoeff[key]!s}\n")

    def save_param(self, filename="param.csv"):
        # Save parameters
        maxval = self.beta[self.a[0]][0]

        with open(filename, "w") as fout:
            sstr = "Feature,Score,"
            for j in range(1, self.max_neighbors + 1):
                sstr = f"{sstr}Neighbor {int(j)}, Neighbor {int(j)} score,"

            sstr = f"{sstr}\n"
            fout.write(sstr)
            for i in range(len(self.a)):
                tmp = [self.featname[self.a[i]], str(self.beta[self.a[i]][0] / maxval)]
                for j in range(1, self.max_neighbors + 1):
                    tmp.extend(
                        (
                            str(self.featname[self.a_neighbors[i][j]]),
                            str(self.a_neighbors_score[i][j]),
                        )
                    )
                sstr = f"{','.join(tmp)}\n"
                fout.write(sstr)

    def _input_data_file(self, file_name: str | Path, output: str | list[str]) -> bool:
        self.x_in, self.y_in, self.featname = input_file(file_name, output=output)
        return True

    def _input_data_list(self, x_in, y_in):
        if isinstance(y_in[0], list):
            msg = "Check your input data"
            raise ValueError(msg)
        self.x_in = np.array(x_in).T
        self.y_in = np.array(y_in).reshape(1, len(y_in))
        return True

    def _input_data_ndarray(self, x_in: npt.NDArray, y_in: npt.NDArray, featname=None):
        if y_in.ndim == 1 and len(y_in) > x_in.shape[0]:
            msg = "If y_in is one-dimensional, it should be of equal length to the number of rows in x_in."
            raise ValueError(msg)
        elif y_in.ndim > 1:
            if y_in.shape[0] != x_in.shape[0]:
                msg = "If y_in is multi-dimensional, it should have the same number of rows as x_in."
                raise ValueError(msg)

        if x_in.ndim != TWO_DIMENSIONAL:
            msg = "x_in should be a two-dimensional, sample-by-feature array."
            raise ValueError(msg)
        return self._set_obj_data(x_in, y_in, featname)

    def _input_data_dataframe(
        self,
        df: pd.DataFrame,
        output: pd.Series | npt.NDArray | list[str] | str = "class",
        featname: list[str] | npt.NDArray | pd.Series | None = None,
    ):
        # featname = pd.Series(featname) if not isinstance(featname, pd.Series) else featname
        match output:
            case str():
                if output in df.columns:
                    y_in = df.loc[:, output].to_numpy()
                else:
                    msg = f"{output} was not found as a column in the passed dataframe"
                    raise KeyError(msg)
            case list() | pd.Series() | np.ndarray():
                if len(output) != df.shape[0]:
                    logger.exception(
                        f"The output does not contain an entry for every row of the dataframe. {output} has {len(output)} items while the dataframe has {df.shape[0]}"
                    )
                else:
                    y_in = np.array(output)
            case _:
                msg = f"output is of type {type(output)} and I don't know what to do with that."
                raise TypeError(msg)

        if featname is not None:
            missing_features = featname[~featname.isin(df.columns)]
            if any(missing_features):
                logger.warning(f"{', '.join(missing_features) } were not found in the data")
            featname = df.columns.intersection(featname).to_list()
            x_in = df.loc[:, featname].to_numpy()
        else:
            x_in = df.drop(columns=output).to_numpy()
            featname = df.drop(columns=output).columns.to_list()

        if any(pd.isnull(y_in)):
            msg = "Found null values in output. Remove or fix these and try again."
            raise ValueError(msg)

        return self._set_obj_data(x_in, y_in, featname)

    # TODO Rename this here and in `_input_data_ndarray` and `_input_data_dataframe`
    def _set_obj_data(self, x_in, y_in, featname):
        self.x_in = x_in.T
        self.y_in = y_in.reshape(1, len(y_in))
        self.featname = featname

    def _check_shape(self):
        _, x_col_len = self.x_in.shape
        _, y_col_len = self.y_in.shape
        # if y_row_len != 1:
        #    raise ValueError("Check your input data")
        if x_col_len != y_col_len:
            msg = "The number of samples in input and output should be same"
            raise ValueError(msg)
        return True

    def _permute_data(self, seed=None):
        np.random.seed(seed)
        n = self.x_in.shape[1]

        perm = np.random.permutation(n)
        self.x_in = self.x_in[:, perm]
        self.y_in = self.y_in[:, perm]
