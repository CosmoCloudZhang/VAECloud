import argparse
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import UTILITY


if __name__ == '__main__':
    PARSE = argparse.ArgumentParser(description='Value Double')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    PARSE.add_argument('--covariance', type=str, default='full', help='gauss or full')
    PARSE.add_argument('--sample-size', type=int, default=1024, help='Comparison size if PCA index is absent')
    PARSE.add_argument('--seed', type=int, default=1, help='Subset seed')
    ARG = PARSE.parse_args()
    OUTPUT = UTILITY.main(ARG.tag, ARG.folder, mask_kind='DOUBLE', covariance=ARG.covariance, sample_size=ARG.sample_size, seed=ARG.seed)
