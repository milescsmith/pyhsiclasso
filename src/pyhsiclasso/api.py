#!/usr/bin/env python
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance
from icecream import ic

from pyhsiclasso.hsic_lasso import compute_kernel, hsic_lasso
from pyhsiclasso.input_data import input_file
from pyhsiclasso.nlars import nlars
from pyhsiclasso.plot_figure import plot_dendrogram, plot_heatmap, plot_path


class HSICLasso:
    def __init__(self):
        self.input_file = None
        self.X_in = None
        self.Y_in = None
        self.X = None
        self.Xty = None
        self.path = None
        self.beta = None
        self.A = None
        self.A_neighbors = None
        self.A_neighbors_score = None
        self.lam = None
        self.featname = None
        self.linkage_dist = None
        self.hclust_featname = None
        self.hclust_featnameindex = None
        self.max_neighbors = 10

    def input(self, *args, **kwargs):
        if "output_list" in kwargs:
            output_list = kwargs["output_list"]
            del kwargs["output_list"]
        else:
            output_list = ["class"]

        self._check_args(args)
        if isinstance(args[0], str | Path):
            self._input_data_file(args[0], output_list)
        elif isinstance(args[0], np.ndarray):
            if "featname" in kwargs:
                featname = kwargs["featname"]
                del kwargs["featname"]
            else:
                featname = [f"{int(x)}" for x in range(1, args[0].shape[1] + 1)]

            if len(args) == 2:
                self._input_data_ndarray(args[0], args[1], featname)
            if len(args) == 3:
                self._input_data_ndarray(args[0], args[1], args[2])
        elif isinstance(args[0], pd.DataFrame):
            df = args[0]
            if "featname" in kwargs:
                featname = kwargs["featname"]
                del kwargs["featname"]
            else:
                featname = df.columns.drop(output_list)
            self._input_data_dataframe(
                df,
            )

        if self.X_in is None or self.Y_in is None:
            msg = "Check your input data"
            raise ValueError(msg)
        self._check_shape()
        return True

    def regression(
        self,
        num_feat: int = 5,
        B: int = 20,
        M: int = 3,
        discrete_x: bool = False,
        max_neighbors: int = 10,
        n_jobs: int = -1,
        covars: npt.ArrayLike | None = None,
        covars_kernel: str = "Gaussian",
    ) -> bool:
        covars = np.array([]) if covars is None else covars

        self._run_hsic_lasso(
            num_feat=num_feat,
            y_kernel="Gaussian",
            B=B,
            M=M,
            discrete_x=discrete_x,
            max_neighbors=max_neighbors,
            n_jobs=n_jobs,
            covars=covars,
            covars_kernel=covars_kernel,
        )

        return True

    def classification(
        self,
        num_feat: int = 5,
        B: int = 20,
        M: int = 3,
        discrete_x: bool = False,
        max_neighbors: int = 10,
        n_jobs: int = -1,
        covars: npt.ArrayLike | None = None,
        covars_kernel: str = "Gaussian",
    ) -> bool:

        covars = np.array([]) if covars is None else covars
        self._run_hsic_lasso(
            num_feat=num_feat,
            y_kernel="Delta",
            B=B,
            M=M,
            discrete_x=discrete_x,
            max_neighbors=max_neighbors,
            n_jobs=n_jobs,
            covars=covars,
            covars_kernel=covars_kernel,
        )

        return True

    def _run_hsic_lasso(
        self,
        y_kernel: str,
        num_feat: int,
        B: int,
        M: int,
        discrete_x: bool,
        max_neighbors: int,
        n_jobs: int,
        covars: npt.ArrayLike,
        covars_kernel: str,
    ) -> bool:
        if self.X_in is None or self.Y_in is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)
        self.max_neighbors = max_neighbors
        n = self.X_in.shape[1]
        B = B or n
        x_kernel = "Delta" if discrete_x else "Gaussian"
        numblocks = n / B
        discarded = n % B
        ic(f"Block HSIC Lasso B = {B}.")


        if discarded:
            msg = (
                f"B {B} must be an exact divisor of the number of samples {n}. Number "
                f"of blocks {numblocks} will be approximated to {int(numblocks)}."
            )
            warnings.warn(msg, RuntimeWarning)
            numblocks = int(numblocks)

        # Number of permutations of the block HSIC
        M = 1 + bool(numblocks - 1) * (M - 1)
        ic(f"M set to {M}.")
        additional_text = " and Gaussian kernel for the covariates" if covars.size else ""
        ic(f"Using {x_kernel} kernel for the features, {y_kernel}kernel for the outcomes{additional_text}.")

        X, Xty, Ky = hsic_lasso(
            self.X_in,
            self.Y_in,
            y_kernel,
            x_kernel,
            n_jobs=n_jobs,
            discarded=discarded,
            B=B,
            M=M,
        )

        # np.concatenate(self.X, axis = 0) * np.sqrt(1/(numblocks * M))
        self.X = X * np.sqrt(1 / (numblocks * M))
        self.Xty = Xty * 1 / (numblocks * M)

        if covars.size:
            if self.X_in.shape[1] != covars.shape[0]:
                msg = f"The number of rows in the covars matrix should be {self.X_in.shape[1]!s}"
                raise UnboundLocalError(msg)

            if covars_kernel == "Gaussian":
                Kc = compute_kernel(covars.transpose(), "Gaussian", B, M, discarded)
            else:
                Kc = compute_kernel(covars.transpose(), "Delta", B, M, discarded)
            Kc = np.reshape(Kc, (n * B * M, 1))

            Ky = Ky * np.sqrt(1 / (numblocks * M))
            Kc = Kc * np.sqrt(1 / (numblocks * M))

            betas = np.dot(Ky.transpose(), Kc) / np.trace(np.dot(Kc.T, Kc))
            # ic(betas)
            self.Xty = self.Xty - betas * np.dot(self.X.transpose(), Kc)

        (
            self.path,
            self.beta,
            self.A,
            self.lam,
            self.A_neighbors,
            self.A_neighbors_score,
        ) = nlars(self.X, self.Xty, num_feat, self.max_neighbors)

        return True

    # For kernel Hierarchical Clustering
    def linkage(self, method="ward"):
        if self.A is None:
            msg = "Run regression/classification first"
            raise UnboundLocalError(msg)
        # selected feature name
        featname_index = []
        featname_selected = []
        for i in range(len(self.A) - 1):
            for index in self.A_neighbors[i]:
                if index not in featname_index:
                    featname_index.append(index)
                    featname_selected.append(self.featname[index])
        self.hclust_featname = featname_selected
        self.hclust_featnameindex = featname_index
        sim = np.dot(self.X[:, featname_index].transpose(), self.X[:, featname_index])
        dist = 1 - sim
        dist = np.maximum(0, dist - np.diag(np.diag(dist)))
        dist_sym = (dist + dist.transpose()) / 2.0
        self.linkage_dist = linkage(distance.squareform(dist_sym), method)

        return True

    def dump(self):
        maxval = self.beta[self.A[0]][0]
        results = [
            " HSICLasso : Result ",
            f"| Order | Feature      | Score | Top-{min(5, len(self.beta) - 1)} Related Feature (Relatedness Score)",
        ]
        for i in range(len(self.A)):
            ofs = f"| {i + 1:<5} | {self.featname[self.A[i]]:<12} | {self.beta[self.A[i]][0] / maxval:.3f} |"
            rf = [
                f" {self.featname[nn]:<12} ({ns:.3f})"
                for nn, ns, _ in zip(self.A_neighbors[i][1:], self.A_neighbors_score[i][1:], range(5), strict=False)
            ]
            row = ofs + ",".join(rf)
            results.append(row + " " * max(0, len(results[1]) - len(row)) + "|")

        results[1] = results[1] + " " * max(0, len(row) - len(results[1])) + "|"
        deco = "=" * ((len(results[1]) - len(results[0])) // 2)
        results[0] = deco + results[0] + deco
        ic("\n".join(results))

        # ic("===== HSICLasso : Path ======")
        # for i in range(len(self.A)):
        #    ic(self.path[self.A[i], 1:])
        # return True

    def plot_heatmap(self, filepath="heatmap.png"):
        if self.linkage_dist is None or self.hclust_featname is None or self.hclust_featnameindex is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)
        plot_heatmap(
            self.X_in[self.hclust_featnameindex, :],
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
        if self.path is None or self.beta is None or self.A is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)
        plot_path(self.path, self.A, filepath)
        return True

    def get_features(self):
        index = self.get_index()

        return [self.featname[i] for i in index]

    def get_features_neighbors(self, feat_index=0, num_neighbors=5):
        index = self.get_index_neighbors(feat_index=feat_index, num_neighbors=num_neighbors)

        return [self.featname[i] for i in index]

    def get_index(self):
        return self.A

    def get_index_score(self):
        return self.beta[self.A, -1]

    def get_index_neighbors(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.A) - 1:
            msg = "Index does not exist"
            raise IndexError(msg)

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.A_neighbors[feat_index][1 : (num_neighbors + 1)]

    def get_index_neighbors_score(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.A) - 1:
            msg = "Index does not exist"
            raise IndexError(msg)

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.A_neighbors_score[feat_index][1 : (num_neighbors + 1)]

    def save_HSICmatrix(self, filename="HSICmatrix.csv"):
        if self.X_in is None or self.Y_in is None:
            msg = "Input your data"
            raise UnboundLocalError(msg)

        self.X, self.X_ty = hsic_lasso(self.X_in, self.Y_in, "Gaussian")

        K = np.dot(self.X.transpose(), self.X)

        np.savetxt(filename, K, delimiter=",", fmt="%.7f")

        return True

    def save_score(self, filename="aggregated_score.csv"):
        maxval = self.beta[self.A[0]][0]

        with open(filename, "w") as fout:
            featscore = {}
            featcorrcoeff = {}
            for i in range(len(self.A)):
                HSIC_XY = self.beta[self.A[i]][0] / maxval

                if self.featname[self.A[i]] not in featscore:
                    featscore[self.featname[self.A[i]]] = HSIC_XY

                    corrcoeff = np.corrcoef(self.X_in[self.A[i]], self.Y_in)[0][1]

                    featcorrcoeff[self.featname[self.A[i]]] = corrcoeff

                else:
                    featscore[self.featname[self.A[i]]] += HSIC_XY

                for j in range(1, self.max_neighbors + 1):
                    HSIC_XX = self.A_neighbors_score[i][j]
                    if self.featname[self.A_neighbors[i][j]] not in featscore:
                        featscore[self.featname[self.A_neighbors[i][j]]] = HSIC_XY * HSIC_XX

                        corrcoeff = np.corrcoef(self.X_in[self.A_neighbors[i][j]], self.Y_in)[0][1]

                        featcorrcoeff[self.featname[self.A_neighbors[i][j]]] = corrcoeff
                    else:
                        featscore[self.featname[self.A_neighbors[i][j]]] += HSIC_XY * HSIC_XX

            # Sorting decending order
            featscore_sorted = sorted(featscore.items(), key=lambda x: x[1], reverse=True)

            # Add Pearson correlation for comparison
            fout.write("Feature,Score,Pearson Corr\n")
            for key, val in featscore_sorted:
                fout.write(f"{key},{val!s},{featcorrcoeff[key]!s}\n")

    def save_param(self, filename="param.csv"):
        # Save parameters
        maxval = self.beta[self.A[0]][0]

        with open(filename, "w") as fout:
            sstr = "Feature,Score,"
            for j in range(1, self.max_neighbors + 1):
                sstr = f"{sstr}Neighbor {int(j)}, Neighbor {int(j)} score,"

            sstr = f"{sstr}\n"
            fout.write(sstr)
            for i in range(len(self.A)):
                tmp = [self.featname[self.A[i]], str(self.beta[self.A[i]][0] / maxval)]
                for j in range(1, self.max_neighbors + 1):
                    tmp.extend(
                        (
                            str(self.featname[self.A_neighbors[i][j]]),
                            str(self.A_neighbors_score[i][j]),
                        )
                    )
                sstr = f"{','.join(tmp)}\n"
                fout.write(sstr)

    # ========================================

    def _check_args(self, args):
        if len(args) == 0 or len(args) >= 4:
            msg = "Input as input_file(file_name) or input_data(X_in, Y_in)"
            raise SyntaxError(msg)
        elif len(args) == 1:
            if not isinstance(args[0], str | Path):
                msg = "Invalid arguments. Input as input_file(file_name) or input_data(X_in, Y_in)"
                raise TypeError(msg)
            else:
                filename = args[0] if isinstance(args[0], Path) else Path(args[0])
                if filename.exists() and filename.suffix not in [".csv", ".tsv", ".mat"]:
                    msg = "pyhsiclasso can only read .csv, .tsv .mat input files"
                    raise TypeError(msg)
                if filename.suffix not in [".csv", ".tsv", ".mat"]:
                    msg = f"{filename} does not exist but if it did, pyhsiclasso "
                    "can only read .csv, .tsv .mat input files"
                    raise FileNotFoundError(msg)
                if not filename.exists():
                    msg = f"{filename} cannot be found. Check your file name"
                    raise FileNotFoundError(msg)
        elif len(args) == 2:
            if isinstance(args[0], str):
                msg = "Check arg type"
                raise TypeError(msg)
            elif isinstance(args[0], list):
                if not isinstance(args[1], list):
                    msg = "Check arg type"
                    raise TypeError(msg)
            elif isinstance(args[0], np.ndarray):
                if not isinstance(args[1], np.ndarray):
                    msg = "Check arg type"
                    raise TypeError(msg)
            else:
                msg = "Check arg type"
                raise TypeError(msg)
        elif len(args) == 3:
            if (
                not isinstance(args[0], np.ndarray)
                or not isinstance(args[1], np.ndarray)
                or not isinstance(args[2], list)
            ):
                msg = "Check arg type"
                raise TypeError(msg)

        return True

    def _input_data_file(self, file_name, output_list) -> bool:
        self.X_in, self.Y_in, self.featname = input_file(file_name, output_list=output_list)
        return True

    def _input_data_list(self, X_in, Y_in):
        if isinstance(Y_in[0], list):
            msg = "Check your input data"
            raise ValueError(msg)
        self.X_in = np.array(X_in).T
        self.Y_in = np.array(Y_in).reshape(1, len(Y_in))
        return True

    def _input_data_ndarray(self, X_in, Y_in, featname=None):
        if len(Y_in.shape) == 2:
            msg = "Check your input data"
            raise ValueError(msg)
        self.X_in = X_in.T
        self.Y_in = Y_in.reshape(1, len(Y_in))
        self.featname = featname
        return True

    def _input_data_dataframe(
        self, df: pd.DataFrame, output_list: list[str] | None = None, featname: list[str] | None = None
    ):
        if output_list is None:
            output_list = ["class"]
        X_in, Y_in, featname = input_file(file_name=df, output_list=output_list, featname=featname)
        self.X_in = X_in
        self.Y_in = Y_in
        self.featname = featname
        return True

    def _check_shape(self):
        _, x_col_len = self.X_in.shape
        y_row_len, y_col_len = self.Y_in.shape
        # if y_row_len != 1:
        #    raise ValueError("Check your input data")
        if x_col_len != y_col_len:
            msg = "The number of samples in input and output should be same"
            raise ValueError(msg)
        return True

    def _permute_data(self, seed=None):
        np.random.seed(seed)
        n = self.X_in.shape[1]

        perm = np.random.permutation(n)
        self.X_in = self.X_in[:, perm]
        self.Y_in = self.Y_in[:, perm]
