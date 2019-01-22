import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

class ClosedForm:
    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        X_transp = X_train.T
        self.w = np.dot(
            np.linalg.inv(
                np.dot(X_transp, X_train)
            ),
            np.dot(X_transp, Y_train)
        )

    def predict(self, X):
        pass


class GradientDescent:
    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        pass


def main():
    X_train, Y_train = get_XY_train()
    closedForm = ClosedForm()
    gradientDescent = GradientDescent()

    train_models([closedForm, gradientDescent], X_train, Y_train)
    #save_models([closedForm, gradientDescent],
    #            ['ClosedForm.pkl', 'GradientDescent.pkl'])


def get_XY_train():
    files = [
        'training_X_counts.json',
        'training_y.json',
    ]
    XY_train = []
    input_path = project_dir / 'src' / 'features'

    for file in files:
        XY_train.append(pickle.load(open(input_path / file, 'rb')))

    print(XY_train[0])
    print(XY_train[1])

    return XY_train


def train_models(models, X_train, Y_train):
    for model in models:
        model.train(X_train, Y_train)


def save_models(models, filenames):
    output_path = project_dir / 'models'
    for model, name in zip(models, filenames):
        pickle.dump(model, open(output_path / name, 'wb'))


if __name__ == '__main__':
    main()
