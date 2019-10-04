from pipeline.utils import load_config, run_grid_search

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config = load_config(args.config_path)
    run_grid_search(config)


if __name__ == "__main__":
    main()
