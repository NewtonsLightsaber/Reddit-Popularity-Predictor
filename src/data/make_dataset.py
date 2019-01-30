# -*- coding: utf-8 -*-
import logging
import json
from pathlib import Path
from collections import Counter
import nltk
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

project_dir = Path(__file__).resolve().parents[2]

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    Writes the 160 most frequent words in the training set to reports/words.txt.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final datasets from raw data')

    split_data()
    preprocess()
    write_most_freq_words()


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
    Iterate through all 3 interim datasets and preprocess each of them,
    then saving them into 3 files in data/processed/:
        training_data.json
        validation_data.json
        test_data.json
    """
    paths = (project_dir / 'data' / 'interim').glob('*.json')
    for path in paths:
        dataset = get_dataset(path)
        filename = str(path).split('/')[-1]
        preprocess_dataset(dataset)

        output_path = project_dir / 'data' / 'processed'
        with open(output_path / filename, 'w') as fout:
            json.dump(dataset, fout)


def get_dataset(path):
    with open(path) as f:
        dataset = json.load(f)
    return dataset


def preprocess_dataset(dataset):
    """
    Encode is_root and x_counts feature.
    """
    most_freq_words = get_most_freq_words(dataset)

    for data in dataset:
        # Encode is_root feature
        is_root = data['is_root']
        data['is_root'] = 1 if is_root else 0

        # Extract word count feature
        data['x_counts'] = get_x_counts(data, most_freq_words)
        
        # Add Comment Length
        data['comment_length'] = len(data['text'])




    return dataset


def get_most_freq_words(dataset):
    words = [word for data in dataset for word in preprocess_text(data['text'])]
    return [word for (word, _) in Counter(words).most_common(160)]


def get_x_counts(data, most_freq_words):
    x_counts = [0] * 160
    # Stemming as a feature
    # nltk library
    # ref: https://www.nltk.org
    counts = dict(Counter(preprocess_text(stemmer.stem(data['text']))))
    #counts = dict(Counter(preprocess_text(data['text'])))
    for word, count in counts.items():
        if word in most_freq_words:
            x_counts[most_freq_words.index(word)] = count

    return x_counts


def write_most_freq_words():
    training_set = get_dataset(
        project_dir / 'data' / 'interim' / 'training_data.json')
    most_freq_words = get_most_freq_words(training_set)
    with open(project_dir / 'reports' / 'words.txt', 'w') as fout:
        for i, word in enumerate(most_freq_words):
            fout.write('%d. %s\n' % (i+1, word))


def preprocess_text(text):
    return text.lower().split()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
