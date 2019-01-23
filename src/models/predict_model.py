import pickle
from pathlib import Path
from train_model import (
    LinearRegression,
    ClosedForm,
    GradientDescent,
    get_XY_train,
    )

project_dir = Path(__file__).resolve().parents[2]

def main():
    X_train, Y_train = get_XY_train()
    closedForm = get_model(project_dir / 'models' / 'ClosedForm.pkl')
    print(closedForm.rmse(X_train, Y_train))


def get_model(path):
    model = pickle.load(open(path, 'rb'))
    if isinstance(model, LinearRegression):
        return model
    else:
        raise Exception('Invalid model.')


if __name__ == '__main__':
    main()
