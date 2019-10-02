import importlib
from .config_base import ConfigBase
from pipeline.spark.common import read_csv, write_csv
from pipeline.logging.logger import logger
from pipeline.collaborative_models.build_matrix import get_user_item_matrix
import os
import pandas as pd
import pickle
from collections import Counter


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
    transactions = pd.read_csv(config.transactions_path)
    stories = pd.read_csv(config.stories_path)
    users = pd.read_csv(config.customer_path)

    candidates = stories[["customer_id", "story_id", "event_dttm", "event"]]

    feature_extractor = config.feature_extractor

    logger.info("start extract features")
    features = feature_extractor.extract(transactions, stories, users, candidates)

    logger.info("saving data")
    features.to_csv(config.train_data_path, index=False)


def run_train_model(config: ConfigBase):
    train_data = pd.read_csv(config.train_data_path)

    model = config.model()

    logger.info("start fitting model")

    class_to_int = {
        "dislike": 0,
        "skip": 1,
        "view": 2,
        "like": 3
    }

    customers = train_data["customer_id"]
    stories = train_data["story_id"]
    target = [class_to_int[targ] for targ in train_data["event"]]
    time = train_data["event_dttm"]

    train_data.drop(columns=["customer_id", "event_dttm", "story_id", "event"], inplace=True)

    model.fit(train_data, target)

    main_model_path = os.path.join(config.models_path, "main_model.pkl")

    logger.info(f"saving model to {main_model_path}")
    with open(main_model_path, "wb") as f:
        pickle.dump(model, f)


def train_collaborative_model(config: ConfigBase):
    model = config.collaborative_model()

    logger.info("get user item matrix")
    user_item_matrix, user_mapper, item_mapper = get_user_item_matrix(config)

    model.user_mapper = user_mapper
    model.item_mapper = item_mapper

    logger.info("start fitting model")
    model.fit(user_item_matrix)

    model_path = os.path.join(config.collaborative_model_dir, f"{repr(model)}.pkl")

    logger.info(f"saving model to {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def build_inference_data(config):
    logger.info("reading tables")
    transactions = pd.read_csv(config.transactions_path)
    stories = pd.read_csv(config.stories_inference_path)
    users = pd.read_csv(config.customer_path)

    candidates = pd.read_csv(config.stories_inference_path)

    feature_extractor = config.feature_extractor

    logger.info("start extract features")
    features = feature_extractor.extract(transactions, stories, users, candidates)

    logger.info("saving data")
    features.to_csv(config.inference_data, index=False)


def run_predict(config: ConfigBase):
    logger.info("read inference data")

    inference_data = pd.read_csv(config.inference_data)

    customers = inference_data["customer_id"]
    stories = inference_data["story_id"]
    answer_ids = inference_data["answer_id"]

    inference_data.drop(columns=["customer_id", "story_id", "event_dttm", "answer_id"], inplace=True)

    main_model_path = os.path.join(config.models_path, "main_model.pkl")

    logger.info(f"loading model from {main_model_path}")
    with open(main_model_path, "rb") as f:
        model = pickle.loads(f.read())

    logger.info("start predicting")
    prediction = model.predict(inference_data)

    logger.info(f"prediction len = {len(prediction)}")
    logger.info(f"predictions counts: {Counter(prediction).most_common(4)}")

    score_mapper = config.score_mapper

    prediction = [score_mapper[score] for score in prediction]

    result = pd.DataFrame(list(zip(
        answer_ids,
        prediction
    )),
        columns=["answer_id", "score"]
    )

    logger.info(f"result shape {result.shape}")

    test_reaction = pd.read_csv(config.stories_inference_path)[["answer_id"]]

    result_shape = result.shape[0]
    result = test_reaction.merge(result, on="answer_id", how="left").fillna(0)

    if result.shape[0] != result_shape:
        logger.warning(f"mismatching shape in result {result_shape} ans must be {result.shape[0]}")

    result.to_csv(config.submit_data_path, index=False)
