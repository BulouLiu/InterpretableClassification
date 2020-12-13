import logging
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from Models.BaseModel import BaseModel
from utils import utils
import joblib
import os

class DecisionTree(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--splitter", type=str, default="best",
                            help="How to choose split point, best or random")
        parser.add_argument("--max_depth",type=int, default=None,
                            help="Max depth for tree.")
        parser.add_argument("--class_weight",type=str, default="balanced")
        parser.add_argument("--min_samples_leaf",type=int,default=1)
        return BaseModel.parse_model_args(parser)
    
    def __init__(self,args):
        if args.class_weight == "balanced":
            self.model = DecisionTreeClassifier(random_state = args.random_seed, splitter=args.splitter,
                    max_depth=args.max_depth,class_weight=args.class_weight,min_samples_leaf=args.min_samples_leaf)
        else:
            self.model = DecisionTreeClassifier(random_state = args.random_seed, splitter=args.splitter,
                    max_depth=args.max_depth,min_samples_leaf=args.min_samples_leaf)

        self.model_path = "../Checkpoints/DecisionTree/"
        super().__init__(args)
    
    def fit(self, X, y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path,"tree.pkl"))
        joblib.dump(self.model, os.path.join(model_path,"tree.pkl"))
        logging.info('Save model to ' + model_path[:50] + '...')     