# -*- coding: utf-8 -*-
""" Authors: Le Nhat Hung, Negin Ashr """
import logging
import pickle
import json
from pathlib import Path
from models import *
from train_model import get_XY_train, reduce_stem

project_dir = Path(__file__).resolve().parents[2]

def main():
    """
    Prints MSEs of each saved model.
    """
    logger = logging.getLogger(__name__)
    logger.info('making prediction with saved models')

    features_path = project_dir / 'src' / 'features'
    models_path = project_dir / 'models'
    predictions_path = project_dir / 'reports'

    X_train, X_train_160, X_train_60, X_train_no_text, Y_train = get_XY_train(features_path)
    X_validate, X_validate_160, X_validate_60, X_validate_no_text, Y_validate = get_XY_validate(features_path)
    X_test, X_test_160, X_test_60, X_test_no_text, Y_test = get_XY_test(features_path)

    """
    The full model is:
        top 160 words + newly added features included

    We've discovered the full model performs best when
    the stem vector contains 100 elements
    (see notebook `3.0-lnh-task-3-experimentation`)
    """
    optimal_size = 100

    # Reduce stem vector size
    X_train = reduce_stem(X_train, optimal_size)
    X_validate = reduce_stem(X_validate, optimal_size)
    X_test = reduce_stem(X_test, optimal_size)

    filenames = (
        'ClosedForm.pkl',
        'ClosedForm_160.pkl',
        'ClosedForm_60.pkl',
        'ClosedForm_no_text.pkl',
        'GradientDescent.pkl',
        'GradientDescent_160.pkl',
        'GradientDescent_60.pkl',
        'GradientDescent_no_text.pkl',
    )
    closedForm, \
    closedForm160, \
    closedForm60, \
    closedFormNoText, \
    gradientDescent, \
    gradientDescent160, \
    gradientDescent60, \
    gradientDescentNoText = get_models(
        models_path,
        filenames=filenames,
    )

    model_name_pairs = (
        (closedForm, 'ClosedForm'),
        (closedForm160, 'ClosedForm_160'),
        (closedForm60, 'ClosedForm_60'),
        (closedFormNoText, 'ClosedForm_no_text'),
        (gradientDescent, 'GradientDescent'),
        (gradientDescent160, 'GradientDescent_160'),
        (gradientDescent60, 'GradientDescent_60'),
        (gradientDescentNoText, 'GradientDescent_no_text'),
    )
    X_train_list = [X_train, X_train_160, X_train_60, X_train_no_text] * 2
    X_validate_list = [X_validate, X_validate_160, X_validate_60, X_validate_no_text] * 2
    X_test_list = [X_test, X_test_160, X_test_60, X_test_no_text] * 2

    predictions_train = get_predictions(model_name_pairs, X_train_list, Y_train, 'train')
    predictions_validate = get_predictions(model_name_pairs, X_validate_list, Y_validate, 'validate')
    predictions_test = get_predictions(model_name_pairs, X_test_list, Y_test, 'test')

    save_predictions(predictions_train, predictions_path / 'predictions_train.json')
    save_predictions(predictions_validate, predictions_path / 'predictions_validate.json')
    save_predictions(predictions_test, predictions_path / 'predictions_test.json')


def get_predictions(model_name_pairs, X_list, Y, suffix):
    """
    Return a list of dictionaries of structure:
        [
            ... ,
            {
                'name': i.g. ClosedForm_60_train,
                'hparams': {'w_0': ... , 'beta': ... , 'eta_0': ... , 'eps': ... },
                'num_iterations': ... ,
                'y_predicted': ... ,
                'mse': ...
            }
            ...
        ]
    """
    predictions = []
    for (model, name), X in zip(model_name_pairs, X_list):
        print('%s %s:' % (name, suffix))
        print('MSE: %.16f' % model.mse(X, Y))

        if isinstance(model, GradientDescent):
            hparams = model.get_hyperparams()
            #hparams['w_0'] = [w for [w] in hparams['w_0'].tolist()]
            """
            For unknown reasons, the w_0 hparam values are distorted at this step.
            Since all gradient descent models had a zero vector for w_0, we'll
            hardcode it.
            """
            hparams['w_0'] = [0] * hparams['w_0'].shape[0]
        else:
            hparams = None

        prediction = {
            'name': name + '_' + suffix,
            'hparams': hparams,
            'num_iterations': model.num_iterations if isinstance(model, GradientDescent) else None,
            'y_predicted': [y for [y] in model.predict(X).tolist()],
            'mse': model.mse(X, Y),
        }
        predictions.append(prediction)

    return predictions


def save_predictions(predictions, path):
    logger = logging.getLogger(__name__)
    json.dump(predictions, open(path, 'w'))
    logger.info('saved predictions to %s' % path)


def get_XY_validate(features_path):
    files = (
        'validation_X.pkl',
        'validation_X_160.pkl',
        'validation_X_60.pkl',
        'validation_X_no_text.pkl',
        'validation_y.pkl',
    )
    XY_validate = []

    for file in files:
        XY_validate.append(pickle.load(open(features_path / file, 'rb')))

    return XY_validate


def get_XY_test(features_path):
    files = (
        'test_X.pkl',
        'test_X_160.pkl',
        'test_X_60.pkl',
        'test_X_no_text.pkl',
        'test_y.pkl',
    )
    XY_test = []

    for file in files:
        XY_test.append(pickle.load(open(features_path / file, 'rb')))

    return XY_test


def get_models(models_path, filenames=None):
    models = []
    if filenames is None:
        models = None
    else:
        for name in filenames:
            path = models_path / name
            model = get_model(path)
            models.append(model)

    return models


def get_model(path):
    model = pickle.load(open(path, 'rb'))
    if isinstance(model, LinearRegression):
        return model
    else:
        raise Exception('Invalid model.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
