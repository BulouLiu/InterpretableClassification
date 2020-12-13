import os
import numpy as np

class DataLoader:
    @staticmethod
    def parse_loader_args(parser):
        parser.add_argument('--data_path', type=str, default="../data",
                            help='Saving path for data set.')
        parser.add_argument('--feature_num', type=int, default=42,
                            help='Number of features.')
        return parser
    
    def __init__(self, args):
        self.data_path = args.data_path
        self.feature_num = args.feature_num
    
    def _load_data(self):
        self.train_feature = []
        self.train_label = []
        self.test_feature = []
        self.test_label = []
        fold_l = ['0', '1', '2', '3', '4']
        for i in range(5):
            self.test_feature.append(np.loadtxt(os.path.join(self.data_path, fold_l[i] , 'feature.txt')))
            self.test_label.append(np.loadtxt(os.path.join( self.data_path, fold_l[i], 'label.txt')).astype(int))
        for i in range(5):
            tmp_feature = np.zeros(shape=(0, self.feature_num))
            tmp_label = np.zeros(shape=(0, ))
            for j in range(5):
                if j != i:
                    tmp_feature = np.concatenate((tmp_feature, self.test_feature[j]), axis=0)
                    tmp_label = np.concatenate((tmp_label, self.test_label[j]), axis=0)
            self.train_feature.append(tmp_feature)
            self.train_label.append(tmp_label)