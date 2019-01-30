# -*- coding: utf-8 -*-
import logging
import json
import pickle
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

def main():
    logger = logging.getLogger(__name__)
    logger.info('extracting features and targets from processed data')

    build_features()


def build_features():
    """
    For each set (training, validation, test), save the following versions:
        1. 'no_text': no text features
        2. '60': top 60 words
        3. '160': top 160 words (basic, with stop words like 'the' or 'a' included)
        4. Top 160 words + newly added features included
    """
    output_path = project_dir / 'src' / 'features'
    paths = (project_dir / 'data' / 'processed').glob('*.json')
    for path in paths:
        file_prefix = get_file_prefix(path)
        dataset = get_dataset(path)
        X_suffix_pairs = (
            (get_X_160(dataset), '_160'),
            (get_X_60(dataset), '_60'),
            (get_X_no_text(dataset), '_no_text'),
            (get_X_full(dataset), ''),
        )
        for X, file_suffix in X_suffix_pairs:
            save_X(X, output_path, file_prefix, file_suffix)

        save_Y(get_Y(dataset), output_path, file_prefix)


def get_file_prefix(path):
    return str(path).split('/')[-1].split('_')[0]


def get_dataset(path):
    with open(path) as f:
        dataset = json.load(f)
    return dataset


def get_Y(dataset):
    y = []
    for data in dataset:
        y.append([data['popularity_score']])

    y = np.array(y)
    return y


def get_X_160(dataset):
    X = []
    bias_term = 1

    for data in dataset:
        X.append(
              [data['is_root']]
            + [data['controversiality']]
            + [data['children']]
            + data['x_counts']
            + [bias_term])

    X = np.array(X)
    return X


def get_X_60(dataset):
    X = []
    bias_term = 1

    for data in dataset:
        X.append(
              [data['is_root']]
            + [data['controversiality']]
            + [data['children']]
            + data['x_counts'][:60]
            + [bias_term])

    X = np.array(X)
    return X


def get_X_no_text(dataset):
    X = []
    bias_term = 1

    for data in dataset:
        X.append(
              [data['is_root']]
            + [data['controversiality']]
            + [data['children']]
            + [bias_term])

    X = np.array(X)
    return X


def get_X_full(dataset):
    X = []
    bias_term = 1

    for data in dataset:
        X.append(
              [data['is_root']]
            + [data['controversiality']]
            + [data['children']]
            + data['x_counts']
            + [bias_term])

    X = np.array(X)
    return X


def save_X(X, output_path, file_prefix, file_suffix):
    with open(output_path / (file_prefix+'_X'+file_suffix+'.pkl'), 'wb') as fout:
        pickle.dump(X, fout)


def save_Y(Y, output_path, file_prefix):
    with open(output_path / (file_prefix+'_y.pkl'), 'wb') as fout:
        pickle.dump(Y, fout)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
