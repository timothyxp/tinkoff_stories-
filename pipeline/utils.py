import importlib
from .config_base import ConfigBase
from pipeline.spark.common import read_csv, write_csv
from pipeline.logging.logger import logger
from pipeline.collaborative_models.build_matrix import get_user_item_matrix
import os
import pandas as pd
import pickle


def _load_cls(module_path, cls_name):
    module_path_fixed = module_path
    if module_path_fixed.endswith(".py"):
        module_path_fixed = module_path_fixed[:-3]
    module_path_fixed = module_path_fixed.replace("/", ".")
    module = importlib.import_module(module_path_fixed)
    assert hasattr(module, cls_name), "{} file should contain {} class".format(module_path, cls_name)

    cls = getattr(module, cls_name)
    return cls


def load_config(config_path: str):
    return _load_cls(config_path, "Config")()


def load_predict_config(config_path: str):
    return _load_cls(config_path, "PredictConfig")()


def run_train(config: ConfigBase):
    logger.info("reading tables")
    transactions = read_csv(config.transactions_path)
    stories = read_csv(config.stories_path)
    users = read_csv(config.customer_path)

    feature_extractor = config.feature_extractor

    logger.info("start extract features")
    features = feature_extractor.extract(transactions, stories, users)

    logger.info("saving data")
    write_csv(features, config.train_data_path)


def run_train_model(config: ConfigBase):
    train_data = pd.read_csv(config.train_data_path)

    model = config.model()

    logger.info("start fitting model")
    model.fit(train_data)

    main_model_path = os.path.join(config.models_path, "main_model.pkl")
    logger.info(f"saving model to {main_model_path}")
    with open(main_model_path, "rb") as f:
        pickle.dump(model, f)


def train_collaborative_model(config: ConfigBase):
    model = config.collaborative_model()

    logger.info("get user item matrix")
    user_item_matrix, user_mapper, item_mapper = get_user_item_matrix(config)

    model.user_mapper = user_mapper
    model.item_mapper = item_mapper

    logger.info("start fitting model")
    model.fit(user_item_matrix)

    model_path = os.path.join(config.collaborative_model_dir, repr(model))

    logger.info("saving model")
    with open(model_path, "wb") as f:
        pickle.dumps(model, f)


def run_predict(config):
    pass
