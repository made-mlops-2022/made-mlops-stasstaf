import pickle
from typing import Dict

import numpy as np
import pandas as pd

from entities import TrainingParams
from models.mappings import metrics, get_models, ClassifierModelType


def model_fit(features: pd.DataFrame, target: pd.DataFrame, train_params: TrainingParams) -> ClassifierModelType:
    classifiers = get_models(train_params)
    model_type = train_params.model_type
    if model_type not in classifiers:
        raise NotImplementedError(f"{model_type} not supported")
    model = classifiers[model_type]
    model.fit(features, target)
    return model


def model_score(predicts: np.ndarray, target: pd.DataFrame, train_params: TrainingParams) -> Dict[str, float]:
    res = dict()
    for metric_name, metric in metrics.items():
        if metric_name in train_params.evaluate_metrics:
            res[metric_name] = metric(predicts, target)
    return res


def serialize_model(model: ClassifierModelType, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(path_to_model: str) -> ClassifierModelType:
    with open(path_to_model, "rb") as f:
        model = pickle.load(f)
    return model


def load_data(path_to_data: str) -> pd.DataFrame:
    return pd.read_csv(path_to_data)


def save_results(data: np.ndarray, output_file: str):
    pd.DataFrame(data, columns=['condition']).to_csv(output_file, index=False)
