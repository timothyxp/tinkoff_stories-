from pipeline.utils import load_config, build_inference_data

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config = load_config(args.config_path)
    build_inference_data(config)


if __name__ == "__main__":
    main()
