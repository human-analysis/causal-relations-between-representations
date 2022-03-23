# config.py
import os
import argparse
import json
import configparser
import utils
import re
from ast import literal_eval as make_tuple


def parse_args():

    parser = argparse.ArgumentParser(description='Your project title goes here')

    parser.add_argument("-c", "--config", "--args-file", dest="config_file", default="args.txt", help="Specify a config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()


    parser.add_argument('--dataset-test', type=str, default=None, help='name of training dataset')
    parser.add_argument('--dataset-train', type=str, default=None, help='name of training dataset')
    parser.add_argument('--total-classes', type=int, help='number of all classes for classification')
    parser.add_argument('--nclasses-a', type=int, default=None, metavar='', help='number of classes for Ad classification')
    parser.add_argument('--nclasses-t', type=int, default=None, metavar='', help='number of classes for Tar classification')
    parser.add_argument('--model-type-e', type=str, default=None, help='type of encoder')
    # parser.add_argument('--model-type-a', type=str, default=None, help='type of network')
    parser.add_argument('--model-type-t', type=str, default=None, help='type of classifier')
    parser.add_argument('--w', type=float, default=None, help='lambda for adversarial loss')
    parser.add_argument('--model-options-e', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--model-options-T', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--loss-type-e', type=str, default=None, help='adversarial loss method')
    parser.add_argument('--loss-type-r', type=str, default=None, help='regression loss method')
    parser.add_argument('--loss-type-t', type=str, default=None, help='classification loss method')
    parser.add_argument('--loss-options-E', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--loss-options-T', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--loss-options-R', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--evaluation-type-t', type=str, default=None, help='evaluation method')
    parser.add_argument('--evaluation-options-T', type=json.loads, default={}, metavar='', help='evaluation-specific parameters, i.e. \'{"topk": 1}\'')
    parser.add_argument('--nunits', type=int, default=None, help='number of units in hidden layers')
    parser.add_argument('--ngpu', type=int, default=None, help='number of gpus to use')
    parser.add_argument('--idx', type=int, default=None, help='causal function idx')
    parser.add_argument('--ntrain', type=int, default=None, help='number of training data')
    parser.add_argument('--npairs', type=int, default=None, help='number of data pairs in one sample')
    parser.add_argument('--ndimsx', type=int, default=None, help='x dimension')
    parser.add_argument('--ndimsy', type=int, default=None, help='y dimension')
    parser.add_argument('--indim', type=int, default=None, help='input dimension')
    parser.add_argument('--outdim', type=int, default=None, help='output dimension')
    parser.add_argument('--batch-size_train', type=int, default=None, help='batch size for training')
    parser.add_argument('--batch-size-test', type=int, default=None, help='batch size for testing')
    parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--nthreads', type=int, default=None, help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=None, help='manual seed for randomness')
    parser.add_argument('--learning-rate-e', type=float, default=None, help='learning rate')
    parser.add_argument('--optim-method-E', type=str, default=None, help='the optimization routine ')
    parser.add_argument('--optim-options-e', type=json.loads, default={}, metavar='', help='optimizer-specific parameters, i.e. \'{"lr": 0.001}\'')
    parser.add_argument('--scheduler-method-e', type=str, default=None, help='cosine, step, exponential, plateau')
    parser.add_argument('--scheduler-options-e', type=json.loads, default={}, metavar='', help='optimizer-specific parameters')
    parser.add_argument('--r', type=int, default=None, help='embedding dimension')
    parser.add_argument('--sigma', type=float, default=None, help='gaussian kernel parameter')
    parser.add_argument('--reg_proj', type=float, default=None, help='regularization parameter for projection')
    parser.add_argument('--reg', type=float, default=None, help='regularization parameter for regression')


    if os.path.exists(args.config_file):
        config = configparser.ConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items("Arguments"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    return args
