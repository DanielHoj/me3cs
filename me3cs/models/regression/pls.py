from abc import ABC

import numpy as np

from me3cs.misc.handle_data import transform_array_1d_to_2d


class BasePLS(ABC):
    """
    Base class for Partial Least Squares regression.

    Parameters
    ----------
    x : np.ndarray
        The predictor variable data of shape (n_samples, n_features).
    y : np.ndarray
        The response variable data of shape (n_samples, n_response).
    n_components : int, optional
        The number of components to compute. Default is 10.

    Attributes
    ----------
    x : np.ndarray
        The predictor variable data of shape (n_samples, n_features).
    y : np.ndarray
        The response variable data of shape (n_samples, n_response).
    n_components : int
        The number of components to compute.
    x_weight : np.ndarray
        The weight coefficients of the predictor variables of shape (n_features, n_components).
    x_scores : np.ndarray
        The scores of the predictor variables of shape (n_samples, n_components).
    x_loadings : np.ndarray
        The loadings of the predictor variables of shape (n_features, n_components).
    x_loadings_orthogonal : np.ndarray
        The orthogonal loadings of the predictor variables of shape (n_features, n_components).
    y_loadings : np.ndarray
        The loadings of the response variables of shape (n_response, n_components).
    y_scores : np.ndarray
        The scores of the response variables of shape (n_samples, n_components).
    reg : np.ndarray
        The regression coefficients of shape (n_features, n_response).

    Methods
    -------
    fit()
        Fits the PLS model to the data.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, n_components: int = 10) -> None:
        """
        Initializes the `BasePLS` class with the given predictor variable `x`, response variable `y`,
        and the number of components to compute `n_components`.
        """
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
        """
        Fits the PLS model to the data.
        """
        pass


class NIPALS(BasePLS):
    """
    Class for performing Nonlinear Iterative Partial Least Squares (NIPALS) algorithm
    for PLS regression.

    Parameters
    ----------
    x : np.ndarray
        Input matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target matrix of shape (n_samples,) or (n_samples, n_targets).
    n_components : int, optional
        Number of PLS components to compute, by default 10.

    Attributes
    ----------
    x : np.ndarray
        Input matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target matrix of shape (n_samples,) or (n_samples, n_targets).
    n_components : int
        Number of PLS components to compute.
    x_weight : np.ndarray
        Weights for x matrix of shape (n_features, n_components).
    x_scores : np.ndarray
        Scores for x matrix of shape (n_samples, n_components).
    x_loadings : np.ndarray
        Loadings for x matrix of shape (n_features, n_components).
    x_loadings_orthogonal : np.ndarray
        Orthogonal loadings for x matrix of shape (n_features, n_components).
    y_loadings : np.ndarray
        Loadings for y matrix of shape (n_targets, n_components).
    y_scores : np.ndarray
        Scores for y matrix of shape (n_samples, n_components).
    reg : np.ndarray
        Regression coefficients of shape (n_features, n_targets, n_components).

    Methods
    -------
    fit()
        Fit the NIPALS algorithm to the data and compute the PLS components.

    Notes
    -----
    This class inherits from BasePLS class.
    """
    def fit(self) -> None:
        """
        Fit the NIPALS model.
        """
        # Get data
        x = self.x.copy()
        y = transform_array_1d_to_2d(self.y.copy())

        # Assert if PLS1 or PLS2
        if y.shape[1] == 1:
            algo_type = "NIPALS1"
        else:
            algo_type = "NIPALS2"

        for a in range(self.n_components):
            # Initialize weights and scores
            weights = np.random.randn(x.shape[1], 1)
            scores = np.zeros((x.shape[0], 1))
            norm_diff = 1

            while norm_diff > 1e-6:
                # Calculate loadings
                loadings = x.T @ scores / (scores.T @ scores)

                # Normalize loadings and weights
                loadings /= np.sqrt(loadings.T @ loadings)
                weights = x @ loadings / (loadings.T @ loadings)

                # Calculate scores
                new_scores = x @ weights

                # Check convergence
                norm_diff = np.linalg.norm(new_scores - scores)
                scores = new_scores

            # Calculate loadings and scores for y
            y_loadings = y.T @ scores / (scores.T @ scores)
            y_scores = y @ y_loadings

            # Store results
            self.x_weight[:, a] = weights.flatten()
            self.x_scores[:, a] = scores.flatten()
            self.x_loadings[:, a] = loadings.flatten()
            self.y_loadings[:, a] = y_loadings.flatten()
            self.y_scores[:, a] = y_scores.flatten()

            # Deflate X and Y
            x -= scores @ weights.T
            y -= y_scores @ y_loadings.T


class SIMPLS(BasePLS):
    """
    Perform SIMPLS (Simple Partial Least Squares) analysis. To calculate the results, the 'fit()' method
    needs to be called.

    Parameters
    ----------
    x : numpy.ndarray
        The predictor variables.
    y : numpy.ndarray
        The response variables.
    n_components : int, optional
        Number of components to use (default is 10).

    Attributes
    ----------
    x : numpy.ndarray
        The predictor variables.
    y : numpy.ndarray
        The response variables.
    n_components : int
        Number of components to use.
    x_weight : numpy.ndarray
        The x weights for each component.
    x_scores : numpy.ndarray
        The x scores for each component.
    x_loadings : numpy.ndarray
        The x loadings for each component.
    x_loadings_orthogonal : numpy.ndarray
        The orthogonal x loadings for each component.
    y_loadings : numpy.ndarray
        The y loadings for each component.
    y_scores : numpy.ndarray
        The y scores for each component.
    reg : numpy.ndarray
        The regression matrix.

    Methods
    -------
    fit()
        Fit the SIMPLS model.

    Notes
    -----
    SIMPLS is a variant of PLS that simplifies the algorithm by using the
    covariance between the predictors and response variables to calculate the
    weights and loadings. It is commonly used in multivariate regression and
    classification.

    References
    ----------
    1. Wold, Svante, et al. "Principal component analysis." Chemometrics and
       intelligent laboratory systems 2.1-3 (1987): 37-52.
    2. de Jong, Sijmen. "SIMPLS: an alternative approach to partial least
       squares regression." Chemometrics and Intelligent Laboratory Systems
       18.3 (1993): 251-263.
    """
    def fit(self) -> None:
        """
        Fit the SIMPLS model.
        """
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
