from pipeline.utils import load_config, run_train

from pipeline.spark.context import setup_context
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    setup_context()
    config = load_config(args.config_path)
    run_train(config)


if __name__ == "__main__":
    main()
