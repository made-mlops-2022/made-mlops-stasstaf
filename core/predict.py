import logging
import sys

import click

from models.fit_predict import load_model, load_data, save_results

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict(path_to_model: str, path_to_data: str, output_file: str):
    logger.info("trying to load model")
    clf = load_model(path_to_model)
    logger.info('model loaded')
    logger.info("trying to load data")
    data = load_data(path_to_data)
    logger.info('data loaded')
    logger.info('starting model inference')
    res = clf.predict(data)
    logger.info('inference done')
    logger.info('saving results')
    save_results(res, output_file)
    logger.info(f'results saved to {output_file}')


@click.command(name="predict")
@click.argument("path_to_model")
@click.argument("path_to_data")
@click.argument("output_file")
def predict_command(path_to_model: str, path_to_data: str, output_file: str):
    predict(path_to_model, path_to_data, output_file)


if __name__ == "__main__":
    predict_command()
