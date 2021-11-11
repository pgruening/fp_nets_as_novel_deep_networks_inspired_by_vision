import argparse


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model_folder', type=str, default=None)

    return parser.parse_args()


def run(options):
    pass
