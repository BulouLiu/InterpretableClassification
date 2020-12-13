import logging
import numpy as np
from sklearn.neural_network import MLPClassifier

from Models.BaseModel import BaseModel
from utils import utils


class MLP(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        self.model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, ), random_state=1, max_iter=500,
                          early_stopping=True, validation_fraction=0.1, activation='tanh')
        super().__init__(args)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        logging.info('Save model to ' + model_path[:50] + '...')