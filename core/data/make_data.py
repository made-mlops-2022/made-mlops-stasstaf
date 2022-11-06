from typing import Tuple
import pandas as pd
from entities import TrainingPipelineParams
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
        data: pd.DataFrame,
        params: TrainingPipelineParams) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_name = params.feature_params.target_col
    x = data.drop(target_name, axis=1)
    y = data[target_name]
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=params.splitting_params.val_size,
        random_state=params.splitting_params.random_state,
        stratify=data[params.feature_params.target_col] if params.splitting_params.stratify else None
    )
    return x_train, x_val, y_train, y_val
