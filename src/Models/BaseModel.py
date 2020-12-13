import numpy as np

from utils import utils

class BaseModel:
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--model_path",type=str, default="../Checkpoints",help="Path to save model.")
        parser.add_argument('--metrics', type=str, default='["f1","accuracy","Recall","Precision"]',
                            help='Evaluation metrics')
        return parser
    
    def __init__(self,args):
        self.feature_num = args.feature_num
        self.metrics = [m.strip().lower() for m in eval(args.metrics)]
        self.model_path = args.model_path

    def predict(self,X):
        pass

    def evaluate(self, X, y):
        predictions = self.predict(X)
        labels = y
        return utils.evaluate_method(predictions, labels, self.metrics)
    
    def print_res(self, X, y):
        """
        Construct the final test result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(X, y)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        return res_str