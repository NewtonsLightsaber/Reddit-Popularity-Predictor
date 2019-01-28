import pickle
from pathlib import Path
from models import *
from train_model import get_XY_train

project_dir = Path(__file__).resolve().parents[2]

def main():
    """
    Prints MSEs of each saved model.
    """
    logger = logging.getLogger(__name__)
    logger.info('making prediction with saved models')

    features_path = project_dir / 'src' / 'features'
    models_path = project / 'models'

    X_train, X_train_160, X_train_60, X_train_no_text, Y_train = get_XY_train(features_path)
    X_validate, X_validate_160, X_validate_60, X_validate_no_text, Y_validate = get_XY_validate(features_path)
    models = get_models(models_path)

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


def get_XY_validate(features_path):
    files = (
        'validation_X.pkl',
        'validation_X_160.pkl',
        'validation_X_60.pkl',
        'validation_X_no_text.pkl',
        'validation_y.pkl',
    )
    XY_validate = []
    input_path = get_features_path()

    for file in files:
        XY_validate.append(pickle.load(open(input_path / file, 'rb')))

    return XY_validate


def get_XY_test():
    files = (
        'test_X.pkl',
        'test_X_160.pkl',
        'test_X_60.pkl',
        'test_X_no_text.pkl',
        'test_y.pkl',
    )
    XY_test = []
    input_path = get_features_path()

    for file in files:
        XY_test.append(pickle.load(open(input_path / file, 'rb')))

    return XY_test


def get_models(models_path):
    models = []
    paths = models_path.glob('*.pkl')
    for path in paths:
        models.append(get_model(path))

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
