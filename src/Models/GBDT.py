import logging
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from Models.BaseModel import BaseModel
from utils import utils


class GBDT(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        self.model = model = GradientBoostingClassifier(n_estimators=3000, max_depth=6, min_samples_split=2, learning_rate=0.1)
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