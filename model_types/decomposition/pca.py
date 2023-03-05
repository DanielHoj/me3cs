from dataclasses import dataclass

import numpy as np

from misc.handle_data import transform_array_1d_to_2d

EPS = np.finfo(float).eps
MAX_ITER = 150


@dataclass
class BaseClassPCA:
    x: np.ndarray
    n_components: int

    scores: np.ndarray = None
    loadings: np.ndarray = None
    explained_variance: np.ndarray = None
    cumulative_explained_variance: np.ndarray = None

    def __post_init__(self) -> None:
        self.init_result()

    def init_result(self) -> None:
        self.scores = np.empty((self.x.shape[0], self.n_components))
        self.loadings = np.empty((self.x.shape[1], self.n_components))
        self.explained_variance = np.empty((1, self.n_components))
        self.cumulative_explained_variance = np.empty((1, self.n_components))


@dataclass
class SVD(BaseClassPCA):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.fit()

    def fit(self) -> None:
        x = self.x.copy()
        U, S, V = np.linalg.svd(x, full_matrices=False)
        scores = np.matmul(U, np.diag(S))
        self.loadings = V[: self.n_components, :].T
        self.scores = scores[:, : self.n_components]
        self.explained_variance = np.square(S[: self.n_components]) / np.square(
            S[: self.n_components]
        ).sum(axis=0)


@dataclass
class EigenDecomposition(BaseClassPCA):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.fit()

    def fit(self) -> None:
        x = self.x.copy()
        cov_mat = np.cov(x, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eigen_values)[::-1][: self.n_components]
        sorted_eigenvalues = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        eigenvector_subset = sorted_eigenvectors[:, 0 : self.n_components]

        self.loadings = eigenvector_subset * np.sqrt(sorted_eigenvalues)
        self.scores = np.dot(x, eigenvector_subset)
        self.explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)


@dataclass
class NIPALS(BaseClassPCA):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.fit()

    def fit(self) -> None:
        x_to_deflate = self.x.copy()
        score = transform_array_1d_to_2d(x_to_deflate[:, 0])

        for component in range(self.n_components):
            for i in range(MAX_ITER):
                loading = np.matmul(x_to_deflate.T, score) / np.matmul(score.T, score)
                loading = loading / np.sqrt(np.matmul(loading.T, loading))
                score_old = score
                score = np.matmul(x_to_deflate, loading) / np.matmul(loading.T, loading)
                if np.square((score_old - score)).sum() < EPS:
                    break
            x_to_deflate -= np.matmul(score, loading.T)
            self.scores[:, component] = score.flatten()
            self.loadings[:, component] = loading.flatten()
            self.explained_variance[:, component] = np.matmul(score.T, score)

        self.explained_variance = (
            self.explained_variance / self.explained_variance.sum()
        )


PCA = {"eigen": EigenDecomposition, "svd": SVD, "nipals": NIPALS}
