# -*- coding: utf-8 -*-
import json
import pickle
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

def main():
    paths = (project_dir / 'data' / 'processed').glob('*.json')
    for path in paths:
        file_prefix = get_file_prefix(path)
        dataset = get_dataset(path)
        X, y = get_XY(dataset)
        write_to_file([X, y], file_prefix)


def get_file_prefix(path):
    return str(path).split('/')[-1].split('_')[0]


def get_dataset(path):
    with open(path) as f:
        dataset = json.load(f)
    return dataset


def get_XY(dataset):
    X, y = [], []
    bias_term = 1

    for data in dataset:
        y.append([data['popularity_score']])
        X.append(
            [data['is_root']]
            + [data['controversiality']]
            + [data['children']]
            + data['x_counts']
            + [bias_term])

    X = np.array(X)
    y = np.array(y)
    return X, y


def write_to_file(features, file_prefix):
    output_path = project_dir / 'src' / 'features'
    for feature, suffix in zip(features, ['X', 'y']):
        with open(output_path / (file_prefix+'_'+suffix+'.pkl'), 'wb') as fout:
            pickle.dump(feature, fout)


if __name__ == '__main__':
    main()
