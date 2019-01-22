import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

class LinearRegression:
    w = None

class ClosedForm(LinearRegression):
    def train(self, X_train, Y_train):
        """
        Find weight vector w from the features and labels matrixes
        Side effect: return the weight vector w
        """
        X_transp = X_train.T
        self.w = np.dot(
            np.linalg.inv( np.dot(X_transp, X_train) ),
            np.dot(X_transp, Y_train)
            )

        return self.w

    def predict(self, X):
        """
        Return the prediction vector.
        Side effect: raise Exception if model model isn't trained
                    i.e. self.w is None
        """
        if self.w is not None:
            return np.dot(X, self.w)
        else:
            raise Exception('Model is not trained.')


class GradientDescent(LinearRegression):
    def train(self, X_train, Y_train):
        pass

    def predict(self, X):
        pass


def main():
    X_train, Y_train = get_XY_train()
    closedForm = ClosedForm()
    gradientDescent = GradientDescent()
    filenames = [
        'ClosedForm.pkl',
        'GradientDescent.pkl',
        ]

    train_models([closedForm, gradientDescent], X_train, Y_train)
    save_models([closedForm, gradientDescent], filenames)


def get_XY_train():
    files = [
        'training_X_counts.json',
        'training_y.json',
        ]
    XY_train = []
    input_path = project_dir / 'src' / 'features'

    for file in files:
        XY_train.append(pickle.load(open(input_path / file, 'rb')))

    return XY_train

def train_models(models, X_train, Y_train):
    for model in models:
        model.train(X_train, Y_train)


def save_models(models, filenames):
    output_path = project_dir / 'models'
    for model, name in zip(models, filenames):
        if model.w is not None:
            pickle.dump(model, open(output_path / name, 'wb'))


if __name__ == '__main__':
    main()
