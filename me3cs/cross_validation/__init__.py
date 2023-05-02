from typing import Union


from .cross_validation_types import cross_validation_types

TYPING_CV_STR = tuple(list(cross_validation_types.keys()))
TYPING_CV_STR = Union[TYPING_CV_STR]
TYPING_CV = tuple(list(cross_validation_types.values()))
TYPING_CV = Union[TYPING_CV]
