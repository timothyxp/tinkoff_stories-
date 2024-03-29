import importlib
from .config_base import ConfigBase
from pipeline.spark.common import read_csv, write_csv
from pipeline.logging.logger import logger
from pipeline.collaborative_models.build_matrix import get_user_item_matrix
import os
import pandas as pd
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from itertools import product
from pipeline.metrics.custom_tinkoff import tinkoff_custom
from catboost import CatBoostClassifier
import json



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
    train_data = pd.read_csv(config.train_data_path, low_memory=False)

    cat_features = []

    drop_columns = ["customer_id", "event_dttm", "story_id", "event"]

    for column in train_data.dtypes.keys():
        typ = str(train_data.dtypes[column])
        if "int" in typ or "float" in typ or "bool" in typ or column in drop_columns:
            continue

        logger.debug(f"cat column {column}")
        cat_features.append(column)
        train_data[column] = train_data[column].astype(str)

    logger.debug(f"have {repr(cat_features)} columns")
    with open("cat_festures.json", "w") as f:
        f.write(json.dumps(cat_features))

    model = CatBoostClassifier(
        learning_rate=0.09,
        max_depth=5,
        iterations=90,
        thread_count=8,
        cat_features=cat_features
    )

    logger.info("start fitting model")

    customers = train_data["customer_id"]
    stories = train_data["story_id"]
    target = [config.class_to_int[targ] for targ in train_data["event"]]
    time = train_data["event_dttm"]

    train_data.drop(columns=drop_columns, inplace=True)

    model.fit(train_data, target)

    main_model_path = os.path.join(config.models_path, "main_model.pkl")

    logger.info(f"saving model to {main_model_path}")
    with open(main_model_path, "wb") as f:
        pickle.dump(model, f)


def run_grid_search(config: ConfigBase):

    transactions = pd.read_csv(config.transactions_path)
    users = pd.read_csv(config.customer_path)
    train_data = pd.read_csv(config.stories_path)

    #train_data = pd.read_csv(config.train_data_path)

    train_data = train_data.sort_values(by="event_dttm")

    target = [config.class_to_int[targ] for targ in train_data["event"]]
    #train_data.drop(columns=["event"], inplace=True)
    feature_extractor = config.feature_extractor

    #train_data.drop(columns=["event", "customer_id", "story_id", "event_dttm"], inplace=True)

    n_estimators = [90, 120, 150]
    learning_rate = [0.09, 0.12, 0.15]
    num_leaves = [4,5,6]

    class_weight_0 = [0.2]
    class_weight_1 = [0.1]
    class_weight_2 = [0.1]
    class_weight_3 = [0.3]

    all_shape = train_data.shape[0]

    divider = 0.7

    train_shape = int(all_shape * divider)

    X_train = train_data[:train_shape]
    X_test = train_data[train_shape:]

    features = feature_extractor.extract(transactions.copy(), X_train.copy(), users.copy(), X_train.copy())
    features_for_test = feature_extractor.extract(transactions.copy(), X_train.copy(), users.copy(), X_test.copy())

    cat_features = []

    ignore_columns = ["customer_id", "event_dttm", "story_id", "event"]

    for column in features.dtypes.keys():
        typ = str(features.dtypes[column])
        if "int" in typ or "float" in typ or "bool" in typ or column in ignore_columns:
            continue

        logger.debug(f"cat column {column}")
        cat_features.append(column)
        features[column] = features[column].astype(str)
        features_for_test[column] = features_for_test[column].astype(str)

    X_train = X_train.merge(features, on=['customer_id', 'story_id', 'event_dttm', 'event'], how='left')
    X_train.drop(columns=ignore_columns, inplace=True)
    X_test = X_test.merge(features_for_test, on=['customer_id', 'story_id', 'event_dttm', 'event'], how='left')
    X_test.drop(columns=ignore_columns, inplace=True)

    Y_train = target[:train_shape]
    Y_test = target[train_shape:]

    logger.info(f"train shape = {X_train.shape[0]}")
    logger.info(f"test shape = {X_test.shape[0]}")

    hyper_parameters = product(n_estimators, learning_rate, num_leaves,
                               class_weight_0, class_weight_1, class_weight_2, class_weight_3)

    best_hyper_params = ()
    best_metric = -1

    for n_estimator, lr, num_leave, cw0, cw1, cw2, cw3  in hyper_parameters:
        logger.info(f"start optimize with params, "
                    f"n_estimators={n_estimator}, "
                    f"learning_rate={lr}, "
                    f"num_leaves={num_leave}"
                    f"cw_0={cw0}, "
                    f"cw_1={cw1}, "
                    f"cw_2={cw2}, "
                    f"cw_3={cw3}"
                    )
        model = CatBoostClassifier(
            iterations=n_estimator,
            max_depth=num_leave,
            learning_rate=lr,
            thread_count=8,
            verbose=0,
            cat_features=cat_features
        )

        logger.debug("fitting")
        model.fit(X_train, Y_train)

        predctions = model.predict(X_test)

        predictions = [config.score_mapper[pred[0]] for pred in predctions]

        metric = tinkoff_custom(predictions, Y_test)

        logger.debug(f"have metric {metric}")

        if metric > best_metric:
            best_metric = metric

            best_hyper_params = (n_estimator, lr, num_leave, cw0, cw1, cw2, cw3)

    logger.info(f"optimize to {best_metric}")
    logger.info(f"best params, "
                f"n_estimators={best_hyper_params[0]}, "
                f"learning_rate={best_hyper_params[1]}, "
                f"num_leaves={best_hyper_params[2]}, "
                f"cw_0={best_hyper_params[3]}, "
                f"cw_1={best_hyper_params[4]}, "
                f"cw_2={best_hyper_params[5]}, "
                f"cw_3={best_hyper_params[6]}"
    )
    logger.info("optimization finished")



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
    stories = pd.read_csv(config.stories_path)
    users = pd.read_csv(config.customer_inference_path)

    candidates = pd.read_csv(config.stories_inference_path)

    feature_extractor = config.feature_extractor

    logger.info("start extract features")
    features = feature_extractor.extract(transactions, stories, users, candidates)

    logger.info(f"features shape {features.shape}")

    logger.info("saving data")
    features.to_csv(config.inference_data, index=False)


def run_predict(config: ConfigBase):
    logger.info("read inference data")

    inference_data = pd.read_csv(config.inference_data, low_memory=False)

    with open("cat_festures.json", "r") as f:
        cat_features = json.loads(f.read())

    for column in cat_features:
        inference_data[column] = inference_data[column].astype(str)

    logger.info(f"inference data shape {inference_data.shape}")

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
    prediction = [pred[0] for pred in prediction]

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
