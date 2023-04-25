import numpy as np
import pandas as pd


def validate_data(data: [np.ndarray | pd.Series | pd.DataFrame]) -> None:
    if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
        raise ValueError("Please input numpy array or pandas dataframe or series")


def save_column_and_index(data: [pd.DataFrame | pd.Series]) -> [dict[list, list] | dict[list]]:
    if isinstance(data, pd.DataFrame):
        return {"index": data.index,
                "columns": data.columns,
                }
    elif isinstance(data, pd.Series):
        return {"index": data.index}


def to_pandas(data: [np.ndarray, pd.Series, pd.DataFrame]):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data
    if data.ndim == 1:
        return pd.Series(data, name="Variable 1")
    else:
        column = [f"Variable {i + 1}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=column)


def bit_type(data: [np.ndarray | pd.Series | pd.DataFrame], bit: str = "float32") -> np.ndarray:
    return np.asfarray(data, dtype=f"{bit}")
