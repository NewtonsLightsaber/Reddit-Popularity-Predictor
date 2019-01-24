import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
from pathlib import Path
import sys

project_dir = Path(__file__).resolve().parents[2]

class LinearRegression:
    w = None

    def is_trained(self):
        return self.w is not None

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


class ClosedForm(LinearRegression):
    """
    Closed form linear regression solution.
    Inherit LinearRegression class.
    """

    def train(self, X_train, Y_train):
        """
        Save weight vector w as field of the instance,
        from the features and labels matrixes.
        Side effect: return weights w
        """
        X_transp = X_train.T
        self.w = np.dot(
            np.linalg.inv( X_transp.dot(X_train) ),
            X_transp.dot(Y_train)
            )
        return self.w

    def rmse(self, X, Y):
        """
        Return the Root Mean Squared Error of the predicted vector
        with regards to the target vector
        """
        n = Y.shape[0]
        return np.linalg.norm(self.predict(X) - Y) / np.sqrt(n)


class gradientDescent(LinearRegression):

    def gradErr(w, X_train, Y_train):
        gradObj = LinearRegression()
        errArg = (Y_train - gradObj.predict(w, X_train)) #(y-Xw)
        finalGradErr = (errArg.T).dot(errArg) #Err = (y-Xw)T * (y-Xw)
        return finalGradErr

    def train(self, X_train, Y_train):
        x = X_train
        epsilon = sys.float_info.epsilon #epsilon
        eta0 = 10 ** (-7)
        beta = 10 ** (-4)
        alpha = eta0 / (1+beta) #steps
        print('alpha = ', alpha)
        wList = []
        i = 0
        wdiff = abs(wList[i+1] - wList[i])
        while(wdiff > epsilon):
            wList[i+1] = wList[i] - ( alpha * (2* x.T.dot(x).dot(wList[i]) - x.T.dot(Y_train))) #or derivative(gradErr)
            print('Generating w, w = ' , wList[i + 1])
            i+=1
            print('and i = ', i)
        print('This is W List' , wList )
        return wList


def main():
    """
    Create, train, and save a closed form solution model and
    a gradient descent model.
    """
    X_train, Y_train = get_XY_train()
    closedForm = ClosedForm()
    gradientDescent = GradientDescent()
    filenames = [
        'ClosedForm.pkl',
        'GradientDescent.pkl',
    ]
    hparams = {
        'w_0': np.zeros((X_train.shape[1], 1)),
        'beta': 1e-4,
        'eta_0': 1e-6,
        'eps': 1e-6,
    }

    closedForm.train(X_train, Y_train)
    #gradientDescent.train(X_train, Y_train, **hparams)
    save_models([closedForm, gradientDescent], filenames)


def get_XY_train():
    files = [
        'training_X.pkl',
        'training_y.pkl',
    ]
    XY_train = []
    input_path = project_dir / 'src' / 'features'

    for file in files:
        XY_train.append(pickle.load(open(input_path / file, 'rb')))

    return XY_train


def save_models(models, filenames):
    output_path = project_dir / 'models'
    for model, name in zip(models, filenames):
        if model.is_trained():
            pickle.dump(model, open(output_path / name, 'wb'))


if __name__ == '__main__':
    main()
