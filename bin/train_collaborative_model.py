from pipeline.utils import load_config, train_collaborative_model

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config = load_config(args.config_path)
    train_collaborative_model(config)


if __name__ == "__main__":
    main()
