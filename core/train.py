import logging
import sys

import click

from data.make_data import read_data, split_train_val_data
from entities import read_training_pipeline_params
from models.fit_predict import model_fit, model_score, serialize_model


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train(config_path: str):
    logger.info("start training")
    training_pipeline_params = read_training_pipeline_params(config_path)
    logger.info("reading data")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info("data read successfully")
    x_train, x_val, y_train, y_val = split_train_val_data(data, training_pipeline_params)
    logger.info("training model")
    model = model_fit(x_train, y_train, training_pipeline_params.train_params)
    logger.info("model trained")
    logger.info("evaluating model")
    predicts = model.predict(x_val)
    metrics = model_score(predicts, y_val, training_pipeline_params.train_params)
    logger.info('eval end')
    logger.info(f'metrics {metrics}')
    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    logger.info(f'model saved to {path_to_model}')


@click.command(name="train")
@click.argument("config_path")
def train_command(config_path: str):
    train(config_path)


if __name__ == "__main__":
    train_command()
