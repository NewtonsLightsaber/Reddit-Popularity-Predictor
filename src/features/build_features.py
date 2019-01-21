# -*- coding: utf-8 -*-
import json
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

def main():
    paths = (project_dir / 'data' / 'processed').glob('*.json')
    for path in paths:
        with open(path) as f:
            dataset = json.load(f)

        X_word_count, y = get_features(dataset)


def build_features(dataset):
    X_word_count = []
    y = []

    for data in dataset:
        X_word_count += [data['word_count']]
        y += data['popularity_score']


if __name__ == '__main__':
    main()
