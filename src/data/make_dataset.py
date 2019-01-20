# -*- coding: utf-8 -*-
import logging
import json
from pathlib import Path


project_dir = Path(__file__).resolve().parents[2]

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    dataset = get_dataset()
    preprocess(dataset)
    split_data(dataset)


def get_dataset():
    with open(project_dir / 'data' / 'raw' / 'proj1_data.json') as f:
        dataset = json.load(f)
    return dataset


def preprocess(dataset):
    for data in dataset:
        is_root = data['is_root']
        data['is_root'] = 1 if is_root else 0


def split_data(dataset):
    output_path = project_dir / 'data' / 'processed'

    files = [
        'training_data.json',
        'validation_data.json',
        'test_data.json',
    ]

    training = dataset[0:10000]
    validation = dataset[10000:11000]
    test = dataset[11000:12000]

    sets = [
        training,
        validation,
        test,
    ]

    for file, set in zip(files, sets):
        with open(output_path / file, 'w') as fout:
            json.dump(set, fout, indent=4)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
