# -*- coding: utf-8 -*-
import logging
import json
from pathlib import Path
from collections import Counter


project_dir = Path(__file__).resolve().parents[2]

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    split_data()
    preprocess()


def split_data():
    output_path = project_dir / 'data' / 'interim'

    files = [
        'training_data.json',
        'validation_data.json',
        'test_data.json',
    ]

    dataset = get_dataset(project_dir / 'data' / 'raw' / 'proj1_data.json')
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
            json.dump(set, fout)


def preprocess():
    """
    Iterate through all 3 interim datasets and preprocess each of them.
    """
    paths = (project_dir / 'data' / 'interim').glob('*.json')
    for path in paths:
        dataset = get_dataset(path)
        filename = str(path).split('/')[-1]
        preprocess_dataset(dataset)

        with open(project_dir / 'data' / 'processed' / filename, 'w') as fout:
            json.dump(dataset, fout)


def get_dataset(path):
    with open(path) as f:
        dataset = json.load(f)
    return dataset


def preprocess_dataset(dataset):
    most_freq_words = get_most_freq_words(dataset)

    for data in dataset:
        # Encode is_root feature
        is_root = data['is_root']
        data['is_root'] = 1 if is_root else 0

        # Split text into lowercase words
        data['text'] = preprocess_text(data['text'])

        # Extract word count feature
        data['word count'] = get_word_count(data, most_freq_words)

    return dataset


def get_most_freq_words(dataset):
    words = [word for data in dataset for word in data['text'].lower().split()]
    return [word for (word, _) in Counter(words).most_common(160)]


def get_word_count(data, most_freq_words):
    word_count = [0] * 160
    counts = dict(Counter(data['text']))
    for word, count in counts.items():
        if word in most_freq_words:
            word_count[most_freq_words.index(word)] = count

    return word_count

def preprocess_text(text):
    return text.lower().split()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
