import os
from . import DATA_PATH
from pipeline.feature_extractors.base import FeatureExtractorBase
from pipeline.collaborative_models.als import ALS
from typing import Callable, Dict


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
            model=None,
            score_mapper: Dict[int, float] = None
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

        customer_iference_path = "customer_test.csv"

        self.customer_inference_path = os.path.join(self.data_path, customer_iference_path)

        if stories_path is None:
            stories_path = "stories_reaction_train.csv"

        self.stories_path = os.path.join(self.data_path, stories_path)

        stories_inference_path = "stories_reaction_test.csv"

        self.stories_inference_path = os.path.join(self.data_path, stories_inference_path)

        if transactions_path is None:
            transactions_path = "transactions.csv"

        self.transactions_path = os.path.join(self.data_path, transactions_path)

        self.data_beautifier = data_beautifier

        self.feature_extractor = feature_extractor

        self.model = model

        self.collaborative_model = collaborative_model

        self.score_mapper = score_mapper

    @property
    def stories_descriptions_parsed(self):
        return f"{self.data_path}/stories_descriptions_parsed.csv"

    @property
    def submit_data_path(self):
        return f"{self.data_path}/submit_result.csv"

    @property
    def inference_data(self):
        return f"{self.data_path}/inference_data.csv"

    @property
    def train_data_path(self):
        return f"{self.data_path}/train_data.csv"

    @property
    def collaborative_model_dir(self):
        return f"{self.models_path}/cf_models"
