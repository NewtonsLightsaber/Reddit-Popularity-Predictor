# -*- coding: utf-8 -*-
""" Authors: Le Nhat Hung, Negin Ashr """
import logging
import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
import sys
from pathlib import Path
from models import *

project_dir = Path(__file__).resolve().parents[2]

def main():
    """
    Create, train, and save a closed form solution model and
    a gradient descent model.
    """
    logger = logging.getLogger(__name__)
    logger.info('training models')

    X_train, Y_train = get_XY_train()
    closedForm = ClosedForm()
    gradientDescent = GradientDescent()
    filenames = [
        'ClosedForm.pkl',
        'GradientDescent.pkl',
    ]
    hparams = {
        'w_0': np.zeros((X_train.shape[1], 1)),
        #'w_0': np.random.rand(X_train.shape[1], 1),
        #'beta': 1e-4, # prof: <1e-3
        'beta': 0,
        'eta_0': 1e-3, # prof: <1e-5
        'eps': 1e-6,
    }

    closedForm.train(X_train, Y_train)
    print('closed form mse: %.16f' % closedForm.mse(X_train, Y_train))
    logger.info('finish closed form model training')
    #print(closedForm.rmse(X_train, Y_train))
    gradientDescent.train(X_train, Y_train, **hparams)
    logger.info('trained gradient descent model')
    print('gradescent mse: %.16f' % gradientDescent.mse(X_train, Y_train))
    save_models([closedForm, gradientDescent], filenames)


def get_XY_train():
    files = [
        'training_X.pkl',
        'training_y.pkl',
    ]
    XY_train = []
    input_path = get_features_path()

    for file in files:
        XY_train += pickle.load(open(input_path / file, 'rb'))

    return XY_train


def save_models(models, filenames):
    output_path = project_dir / 'models'
    for model, name in zip(models, filenames):
        if model.is_trained():
            pickle.dump(model, open(output_path / name, 'wb'))


def get_features_path():
    return project_dir / 'src' / 'features'


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
