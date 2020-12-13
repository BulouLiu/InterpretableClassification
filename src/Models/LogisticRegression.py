import numpy as np
from Models.BaseModel import BaseModel
from utils import utils
import os

import logging

class LogisticRegression(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--init_lr", type=float, default=0.01,help="learning rate.")
        parser.add_argument("--max_iterations", type=int, default=2000, help="Max iteractions.")
        parser.add_argument("--early_stop_patience",type=int, default=20, 
            help="Number of iterations with no loss reduce, then stop training.")
        return BaseModel.parse_model_args(parser)
    
    def __init__(self,args):
        self.random_seed = args.random_seed
        self.feature_num = args.feature_num
        self.class_num = args.class_num
        self.lr = args.init_lr
        self.init_lr = args.init_lr 
        self.max_iterations = args.max_iterations
        self.early_stop_patience = args.early_stop_patience
        self.model_path = "../Checkpoints/LogisticRegression/"
        self._param_init()
        super().__init__(args)

    def _param_init(self):
        np.random.seed(self.random_seed)
        self.weight = np.random.randn(self.class_num, self.feature_num)
        self.bias = np.random.randn(self.class_num)


    def softmax(self,X):
        # X: batch_size * class_num
        r = np.exp(X).T / np.exp(X).sum(axis=1)
        return r.T # batch_size * class_num

    def forward(self, X):
        # X: batch_size * feature_num
        y = self.softmax((np.dot(X , self.weight.T) + self.bias))
        return y
    
    def backward(self, X, y_prob, y_label):
        # X: batch_size * feature_num
        # y_prob: batch_size * class_num
        size = X.shape[0]
        label_dum = np.zeros_like(y_prob)
        label_dum[np.arange(size), y_label.astype(int)+1] = 1.0
        
        loss = - np.sum(label_dum * np.log(y_prob)) / size
        dw = np.dot((y_prob-label_dum).T, X) / size
        db = np.dot((y_prob-label_dum).T, np.ones((size))) / size
        self.weight -= self.lr * dw
        self.bias -= self.lr * db
        return loss
    
    def fit(self, X, y):
        self._param_init()
        last_error = 1e2
        no_desc = 0
        for i in range(self.max_iterations):
            y_prob = self.forward(X)
            loss = self.backward(X, y_prob, y)
            if loss> last_error - 1e-7:
                no_desc += 1
            else:
                no_desc = 0
            if no_desc>=self.early_stop_patience:
                self.lr *= 0.8
                no_desc = 0
                # logging.info("Learning rate adjustment: {}".format(self.lr))
            if self.lr <= self.init_lr*0.1:
                logging.info("Early Stop! [Iter: {}]".format(i))
                break
            last_error = loss
            if i % 500 == 0:
                logging.info("Iter {}: error rate {:.3f}".format(i,loss))

    def predict(self, X):
        return np.argmax(self.forward(X),axis=1)-1
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path+"a")
        np.save(os.path.join(model_path,"Weight"),self.weight)
        np.save(os.path.join(model_path,"Bias"),self.bias)
        logging.info('Save model to ' + model_path[:50] + '...')     