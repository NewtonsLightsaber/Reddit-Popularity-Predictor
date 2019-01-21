import pickle
import numpy as np

project_dir = Path(__file__).resolve().parents[2]

class ClosedForm:
    def __init__(self):
        self.w = np.array

    def train(self, X_train, Y_train):
        pass

    def predict(self, X):
        pass


class GradientDescent:
    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        pass

def main():
    ClosedForm = ClosedForm()
    GradientDescent = GradientDescent()
    ClosedForm.train()
    GradientDescent.train()
    write_to_file(Closed)

def train_models():


    return ClosedForm, GradientDescent

if __name__ == '__main__':
    main()
