import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

from Models.BaseModel import BaseModel
from utils import utils
import joblib
import os

class LR(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--lr", type=float, default=0.01,help="learning rate.")
        parser.add_argument("--max_iterations", type=int, default=500, help="Max iteractions.")
        return BaseModel.parse_model_args(parser)
    
    def __init__(self,args):
        self.model = LogisticRegression(random_state = args.random_seed,
                max_iter=args.max_iterations) 
        self.model_path = "../Checkpoints/LR/"
        super().__init__(args)
    
    def fit(self, X, y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path,"LR.pkl"))
        joblib.dump(self.model, os.path.join(model_path,"LR.pkl"))
        logging.info('Save model to ' + model_path[:50] + '...')     