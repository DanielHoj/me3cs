import numpy as np

from me3cs.misc.metrics import moore_penrose_inverse
from me3cs.models.decomposition.pca import PCA


class PCR:
    def __init__(self, x: np.ndarray, y: np.ndarray, n_components: int) -> None:
        self.x = x
        self.y = y
        self.n_components = n_components
        self.reg = None
        self.x_scores = None
        self.x_loading = None
        self.fit()

    def fit(self) -> None:
        pca = PCA["svd"]
        decomp_model = pca(x=self.x, n_components=self.n_components)
        scores = decomp_model.scores
        loading = decomp_model.loadings
        reg_pca_space = np.dot(moore_penrose_inverse(scores), self.y)
        self.reg = np.dot(loading, reg_pca_space)
        self.x_scores = scores
        self.x_loading = loading
