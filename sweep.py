import os

from main import setup_arg_parser
from trackit.core.boot.sweep import sweep_main


def setup_sweep_arg_parser():
    parser = setup_arg_parser()
    parser.add_argument('--agents_run_limit', type=int, default=0)
    parser.add_argument('--sweep_id', type=str)
    return parser


if __name__ == '__main__':
    parser = setup_sweep_arg_parser()
    args, unknown_args = parser.parse_known_args()
    args.root_path = os.path.dirname(os.path.abspath(__file__))
    sweep_main(args, unknown_args)
