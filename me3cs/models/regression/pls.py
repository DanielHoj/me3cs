from abc import ABC

import numpy as np

from me3cs.misc.handle_data import transform_array_1d_to_2d


class BasePLS(ABC):
    def __init__(self, x: np.ndarray, y: np.ndarray, n_components: int = 10) -> None:
        self.x = x
        self.y = y
        self.n_components = n_components
        self.x_weight = np.ndarray((x.shape[1], n_components))
        self.x_scores = np.ndarray((x.shape[0], n_components))
        self.x_loadings = np.ndarray((x.shape[1], n_components))
        self.x_loadings_orthogonal = np.ndarray((x.shape[1], n_components))
        self.y_loadings = np.ndarray((y.shape[1], n_components))
        self.y_scores = np.ndarray((y.shape[0], n_components))
        self.fit()
        self.reg = np.einsum("ij, kj -> ij", self.x_weight, self.y_loadings).cumsum(
            axis=1
        )

    def fit(self) -> None:
        pass


class NIPALS(BasePLS):
    def fit(self) -> None:
        pass


class SIMPLS(BasePLS):
    def fit(self) -> None:
        # Get data
        x = self.x.copy()
        y = transform_array_1d_to_2d(self.y.copy())

        # Assert if PLS1 or PLS2
        if y.shape[1] == 1:
            algo_type = "SIMPLS1"
        else:
            algo_type = "SIMPLS2"

        # Calculate covariance Matrix
        cov_matrix = x.T @ y

        for a in range(self.n_components):
            # Calculate y weights
            if algo_type == "SIMPLS1":
                y_weights = transform_array_1d_to_2d(np.ones([1]))
            else:
                y_weights = transform_array_1d_to_2d(
                    np.linalg.eigh(cov_matrix.T @ cov_matrix)[1][:, 0]
                )

            x_weights = cov_matrix @ y_weights  # Calculate x weights
            x_scores = x @ x_weights  # Calculate x scores

            # Normalise x scores and weights
            normt = np.sqrt(x_scores.T @ x_scores)
            x_scores = x_scores / normt
            x_weights = x_weights / normt

            x_loadings = x.T @ x_scores  # Calculate x loadings
            y_loadings = y.T @ x_scores  # Calculate y loadings
            y_scores = y @ y_weights  # Calculate y scores
            x_loadings_orthogonal = x_loadings  # initialise orthogonal loadings

            if a > 0:
                x_loadings_orthogonal = (
                        x_loadings_orthogonal
                        - self.x_loadings_orthogonal[:, :a]
                        @ (self.x_loadings_orthogonal[:, :a].T @ x_loadings)
                )  # Orthogonalise x loadings
                y_scores = y_scores - self.x_scores[:, :a] @ (
                        self.x_scores[:, :a].T @ y_scores
                )  # Make y scores perpendicular to previous x scores

            x_loadings_orthogonal = x_loadings_orthogonal / (
                np.sqrt(x_loadings_orthogonal.T @ x_loadings_orthogonal)
            )  # Normalise orthogonal x loadings

            cov_matrix = cov_matrix - x_loadings_orthogonal @ (
                    x_loadings_orthogonal.T @ cov_matrix
            )  # Deflate cov_matrix with respect to current loadings

            self.x_weight[:, a] = x_weights.flatten()
            self.x_scores[:, a] = x_scores.flatten()
            self.x_loadings[:, a] = x_loadings.flatten()
            self.x_loadings_orthogonal[:, a] = x_loadings_orthogonal.flatten()
            self.y_loadings[:, a] = y_loadings.flatten()
            self.y_scores[:, a] = y_scores.flatten()


PLS = {"SIMPLS": SIMPLS,
       "NIPALS": NIPALS,
       }
