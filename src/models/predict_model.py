import pickle
from pathlib import Path
from models import *
from train_model import get_XY_train, get_features_path

project_dir = Path(__file__).resolve().parents[2]

def main():
    """
    Prints MSEs of each saved model.
    """
    logger = logging.getLogger(__name__)
    logger.info('making prediction with saved models')

    X_train, Y_train = get_XY_train()
    X_validate, Y_validate = get_XY_validate()
    models = get_models()

    make_predictions(models, [
        [X_train, Y_train],
        [X_validate, Y_validate],
    ])


def make_predictions(models, datasets):
    set_names = [
        'Training set',
        'Validation set',
        'Test set',
    ]
    for i, model in enumerate(models):
        print('Model %d:' % i)
        for set, name in zip(datasets, set_names):
            print('%s:' % name)
            X, Y = set
            predict(model, X, Y)


def predict(models, X, Y):
    for model in models:
        print(model)
        print('MSE: %.16f' % model.mse(X, Y))


def get_XY_validate():
    files = (
        'validation_X.pkl',
        'validation_y.pkl',
    )
    XY_validate = []
    input_path = get_features_path()

    for file in files:
        XY_validate += pickle.load(open(input_path / file, 'rb'))

    return XY_validate


def get_XY_test():
    files = (
        'test_X.pkl',
        'test_y.pkl',
    )
    XY_test = []
    input_path = get_features_path()

    for file in files:
        XY_test += pickle.load(open(input_path / file, 'rb'))

    return XY_test


def get_models():
    models = []
    paths = (project_dir / models).glob('*.pkl')
    for path in paths:
        models += get_model(path)
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
