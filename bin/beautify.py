from pipeline.utils import load_config

from pipeline.config_base import ConfigBase
import argparse
from pipeline.logging.logger import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    logger.info("load config")
    config: ConfigBase = load_config(args.config_path)

    logger.info("start beautify")

    beautifier = config.data_beautifier
    beautifier.beautify()


if __name__ == "__main__":
    main()
