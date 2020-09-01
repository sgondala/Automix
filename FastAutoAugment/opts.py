import argparse
from theconf import Config as C, ConfigArgumentParser

def parse_search_opts():
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataroot', type=str, default='data/')
    parser.add_argument('--until', type=int, default=5)
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--num-search', type=int, default=200)
    parser.add_argument('--cv-ratio', type=float, default=0.4)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--redis', type=str, default='')
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--num-result-per-cv', type=int, default=200)
    parser.add_argument('--cv-num', type=int, default=1, help='Number of datset splits')
    args = parser.parse_args()

    return args