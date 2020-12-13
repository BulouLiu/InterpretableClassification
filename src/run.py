import os
import sys
import logging
import argparse
import numpy as np


from Models import *
from helpers import DataLoader
from utils import utils

def parse_global_args(parser):
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=2020,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    return parser

def main(args):
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose', 'load', 'buffer']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    np.random.seed(args.random_seed)

    # Read Data
    dataloader = DataLoader.DataLoader(args)
    dataloader._load_data()

    # Define Model
    model = model_name(args)

    # Run Model
    evaluations_list = {}
    for i in range(5):
        model.fit(dataloader.train_feature[i],dataloader.train_label[i])
        evaluations = model.print_res(dataloader.test_feature[i],dataloader.test_label[i])
        evaluation_results = model.evaluate(dataloader.test_feature[i],dataloader.test_label[i])
        for key in evaluation_results:
            if key not in evaluations_list:
                evaluations_list[key] = []
            evaluations_list[key].append(evaluation_results[key])
        logging.info('Test Results at {} times: {}'.format(i,evaluations))
    evaluations_all = {}
    for key in evaluations_list:
        evaluations_all[key] = np.mean(evaluations_list[key])
    logging.info("Average results: {}".format(utils.format_metric(evaluations_all)))

if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='DecisionTree', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = model_name.parse_model_args(parser)
    parser = DataLoader.DataLoader.parse_loader_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.data_path]
    log_file_name = '_'.join(log_args).replace(' ', '_')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main(args)
