# -*- coding: utf-8 -*-
import click
import logging
import json
from pathlib import Path


def main(input_path, output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    files = [
        'training_data.json',
        'validation_data.json',
        'test_data.json',
    ]

    with open(input_path / 'proj1_data.json') as f:
        dataset = json.load(f)

    training = dataset[0:10000]
    validation = dataset[10000:11000]
    test = dataset[11000:12000]

    sets = [
        training,
        validation,
        test,
    ]

    for file, set in zip(files, sets):
        filepath = output_path / file
        if not filepath.is_file():
            with open(filepath, 'w') as fout:
                json.dump(set, fout)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / 'data' / 'raw'
    output_path = project_dir / 'data' / 'processed'

    main(input_path, output_path)
