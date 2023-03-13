from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='visint_incremental', allow_abbrev=False)
    parser.add_argument('--transform', type=str, default='default',
                        help='default or pytorch.')
    parser.add_argument('--featureNet', type=str, default=None,
                        help='feature extractor')
    parser.add_argument('--nt', type=str, default=None,
                        help='task number')
    parser.add_argument('--t_c_arr', type=str, default=None,
                        help='class array for each task')
    parser.add_argument('--seed', type=str, default=None,
                        help='random seed if None')
    parser.add_argument('--validation', type=str, default=False,
                        help='is test with the validation set')
    parser.add_argument('--class_shuffle', type=str, default=False,
                        help='is shuffle the order of classes')
    args = parser.parse_known_args()[0]
    return args