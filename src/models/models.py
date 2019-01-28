# -*- coding: utf-8 -*-
import numpy as np

class LinearRegression:
    """
    Instantiated with a weights vector = None.
    Inhereted by ClosedForm and GradientDescent.
    """
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
    Inherits LinearRegression class.
    """
    def __str__(self):
        return 'Closed form'

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


"""class gradientDescent(LinearRegression):

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
        while True:
            wdiff = abs(wlist[i+1] - wlist[i])
            while(wdiff > epsilon):
                wList[i+1] = wList[i] - ( alpha * (2* (x.T).dot(x).dot(wList[i]) - (x.T).dot(Y_train)) #or derivative(gradErr)
            i = i + 1

        return wList
"""


class GradientDescent(LinearRegression):
    """
    Gradient descent linear regression solution.
    Inherits LinearRegression class.
    """
    w_0, beta, eta_0, eps = [None] * 4

    def __str__(self):
        return ('Gradient descent\n'
                'w_0: %s,\n'
                'beta: %.16f, '
                'eta_0: %.16f, '
                'eps: %.16f.') % (
                    str(self.w_0.tolist()),
                    self.beta,
                    self.eta_0,
                    self.eps,
                )

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
        self.save_hyperparams([w_0, beta, eta_0, eps])

        n = Y_train.shape[0]
        w_prev = w_0
        norm = lambda x : np.linalg.norm(x)
        twoXTX = 2*X_train.T.dot(X_train)
        twoXTy = 2*X_train.T.dot(Y_train)
        i = 1

        #print('w_0: '); print(w_0)
        print('beta: %.16f' % beta)
        print('eta_0: %.16f' % eta_0)

        while True:
            alpha = eta_0 / (1 + beta * i) / n
            grad = twoXTX.dot(w_prev) - twoXTy
            self.w = w_prev - alpha*grad

            wDiff = norm(self.w - w_prev)
            print('i: %d' % i)
            print('wDiff: %.16f' % wDiff)
            print('mse: %.16f' % self.mse(X_train, Y_train))
            if wDiff <= eps:
                break
            else:
                i += 1
                w_prev[:] = self.w

        return self.w


    def save_hyperparams(self, hparams):
        self.w_0, self.beta, self.eta_0, self.eps = hparams
