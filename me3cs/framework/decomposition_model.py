from me3cs.framework.base_model import BaseModel
from me3cs.model_types.decomposition.decomposition_results import DecompositionResults
from me3cs.model_types.decomposition.pca import PCA


class DecompositionModel(BaseModel):
    # TODO: make pca crossvalidation
    def pca(self, algorithm: str = "svd", cross_validation: bool = False) -> None:
        if algorithm not in ["svd", "nipals", "eigen"]:
            raise ValueError(
                f"Please input algorithm as 'svd', 'nipals' or 'eigen'. {algorithm} was passed"
            )
        if not isinstance(algorithm, str):
            raise TypeError("algorithm has to be of type string.")

        # mean center if not mean centered
        if not self.x.preprocessing.data_is_centered:
            if self.options.mean_center:
                self.x.preprocessing.mean_center()

        # Get preprocessed data
        x = self.x.preprocessing.data

        algorithm = PCA[f"{algorithm}"]

        if cross_validation:
            self.__cross_validation_pipeline__(
                algorithm=algorithm, results=DecompositionResults
            )
        else:
            model = algorithm(x=x, n_components=self.options.n_components)
            setattr(
                self,
                "results",
            )

    def mcr(self) -> None:
        # TODO: implement MCR algorithm
        pass

    def __cross_validation_pipeline__(self, algorithm, results):

        # Get called preprocessing
        x_preprocessing = self.x.preprocessing.called
        called_preprocessing = x_preprocessing
