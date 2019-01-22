import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

class LinearRegression:
    w = None

    def is_trained(self):
        return self.w is not None


class ClosedForm(LinearRegression):
    """
    Closed form linear regression solution.
    Inherit LinearRegression class.
    """

    def train(self, X_train, Y_train):
        """
        Save weight vector w as a field of the instance,
        from the features and labels matrixes.
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
        Return the prediction vector y.
        Side effect: raise Exception if model model isn't trained
                    i.e. self.w is None
        """
        if self.w is not None:
            return np.dot(X, self.w)
        else:
            raise Exception('Model is not trained.')

    def rmse(self, X, Y):
        """
        Return the Root Mean Squared Error of the predicted vector
        with regards to the target vector
        """
        n = Y.shape[0]
        return np.linalg.norm(self.predict(X) - Y) / np.sqrt(n)


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
        if model.is_trained():
            pickle.dump(model, open(output_path / name, 'wb'))


if __name__ == '__main__':
    main()
