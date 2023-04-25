from me3cs.missing_data.missing_data import MissingData
from me3cs.framework.data import Data
from me3cs.preprocessing.preprocessing import get_preprocessing_from_dimension


class Branch:
    def __init__(self, data: Data, branches: list) -> None:
        preprocessing_type = get_preprocessing_from_dimension(data.data)

        self.data_class = data
        self._branches = branches

        self.preprocessing = preprocessing_type(data)
        self.missing_data = MissingData(data, self._branches)

    @property
    def data(self):
        return self.data_class.data

    @data.getter
    def data(self):
        return self.data_class.data

    @data.setter
    def data(self, data):
        self.data_class.preprocessing_data.set(data)

    def __repr__(self):
        return f"Data shape: {self.data.shape}\n" \
               f"Preprocessing - {self.preprocessing.called}\n" \
               f""

    def __getitem__(self, key):
        return Branch(self.data[key], self._branches)

    def __array__(self):
        return self.data

    def __len__(self):
        return len(self.data)
