import numpy as np
import pandas as pd


def validate_data(data: [np.ndarray | pd.Series | pd.DataFrame]) -> None:
    """
    Validate the input data to ensure it is a NumPy array, Pandas DataFrame, or Pandas Series.

    This function checks if the input data is an instance of one of the supported types.
    If not, it raises a ValueError with an appropriate error message.

    Parameters
    ----------
    data : np.ndarray, pd.Series, or pd.DataFrame
        The data to be validated.

    Raises
    ------
    ValueError
        If the input data is not an instance of np.ndarray, pd.Series, or pd.DataFrame.
    """
    if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
        raise ValueError("Please input numpy array or pandas dataframe or series")
