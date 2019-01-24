# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
import sys
from pathlib import Path

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

    def mse(self, X, Y):
        """
        Return the Mean Squared Error of the predicted vector
        with regards to the target vector
        """
        return np.square(self.predict(X) - Y).mean()


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
        wList = []
        i = 0
        wdiff = abs(wList[i+1] - wList[i])
        while(wdiff > epsilon):
            wList[i+1] = wList[i] - ( alpha * (2* x.T.dot(x).dot(wList[i]) - x.T.dot(Y_train))) #or derivative(gradErr)
            i+=1
        
        return wList
                                         
                                         
class GradientDescent(LinearRegression):
    def train(self, X_train, Y_train, w_0, beta, eta_0, eps):
        """
        Save weight vector w as field of the instance.
        Inputs:
            X_train: data matrix,
            Y_train: targets,
            w_0: initial weights,
            beta: speed of decay
            eta_0: initial learning rate
            eps: stopping tolerance

        Output:
            Estimated weights w
        """
        X, y = X_train, Y_train
        n = y.shape[0]
        w_prev = w_0
        norm = lambda x : np.linalg.norm(x)
        i = 1

        while True:
            alpha = eta_0 / (1 + beta * i) / n
            grad = X.T.dot(X).dot(w_prev) - X.T.dot(y)
            self.w = w_prev - 2 * alpha * grad

            loss = norm(self.w - w_prev)
            print('loss: %.16f' % loss)
            if loss <= eps:
                break
            else:
                i += 1
                print('i: %d' % i)
                w_prev[:] = self.w

        return self.w


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
        'beta': 1e-7, # prof: <1e-3
        'eta_0': 7e-6, # prof: <1e-5
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
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
