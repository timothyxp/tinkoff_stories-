import os
from . import DATA_PATH
from pipeline.feature_extractors.base import FeatureExtractorBase
from pipeline.collaborative_models.als import ALS
from typing import Callable


class ConfigBase:
    def __init__(
            self,
            experiment_name: str,
            models_path: str = None,
            customer_path: str = None,
            stories_path: str = None,
            transactions_path: str = None,
            data_beautifier = None, #TODO for loading data, descriptions, images and other refs ans parse data
            feature_extractor: FeatureExtractorBase = None, #TODO generate feature, at first simple features
            collaborative_model: Callable[[], ALS] = None,
            model=None
    ):
        self.experiment_name = experiment_name

        self.data_path = os.path.join(DATA_PATH, self.experiment_name)

        if models_path is None:
            models_path = "models"

        self.models_path = os.path.join(self.data_path, models_path)

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.collaborative_model_dir, exist_ok=True)

        if customer_path is None:
            customer_path = "customer_train.csv"

        self.customer_path = os.path.join(self.data_path, customer_path)

        if stories_path is None:
            stories_path = "stories_reaction_train.csv"

        self.stories_path = os.path.join(self.data_path, stories_path)

        if transactions_path is None:
            transactions_path = "transactions.csv"

        self.transactions_path = os.path.join(self.data_path, transactions_path)

        self.data_beautifier = data_beautifier

        self.feature_extractor = feature_extractor

        self.model = model

        self.collaborative_model = collaborative_model

    @property
    def train_data_path(self):
        return f"{self.data_path}/train_data.csv"

    @property
    def collaborative_model_dir(self):
        return f"{self.models_path}/cf_models"


class PredictConfigBase:
    def __init__(
            self,
            experiment_name: str,
            models_path: str = None,
            customer_path: str = None,
            stories_path: str = None,
            transactions_path: str = None,
            result_path: str = None
    ):
        self.experiment_name = experiment_name

        self.data_path = os.path.join(DATA_PATH, self.experiment_name)

        if models_path is None:
            models_path = "models"

        self.models_path = os.path.join(self.data_path, models_path)

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        if customer_path is None:
            customer_path = "customer_test.csv"

        self.customer_path = os.path.join(self.data_path, customer_path)

        if stories_path is None:
            stories_path = "stories_reaction_test.csv"

        self.stories_path = os.path.join(self.data_path, stories_path)

        if transactions_path is None:
            transactions_path = "transactions.csv"

        self.transactions_path = os.path.join(DATA_PATH, transactions_path)

        if result_path is None:
            result_path = "submit.csv"

        self.result_path = os.path.join(DATA_PATH, result_path)
