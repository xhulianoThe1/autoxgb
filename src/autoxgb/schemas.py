import pandas as pd 
from typing import List, Optional, Union, TypeVar

from pydantic import BaseModel

from .enums import ProblemType


class ModelConfig(BaseModel):
    train_filename: Union[TypeVar("pandas.core.frame.DataFrame"), str]
    test_filename: Optional[str] = None
    idx: str
    targets: List[str]
    problem_type: ProblemType
    output: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    categorical_features: List[str]
    num_trials: int
    time_limit: Optional[int] = None
    fast: bool
