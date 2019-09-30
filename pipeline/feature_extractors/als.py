from .base import FeatureExtractorBase
from pipeline.config_base import ConfigBase
import pandas as pd
import os
from pipeline.logging.logger import logger
import pickle


class FeatureExtractorALS(FeatureExtractorBase):
    def __init__(self, config):
        self.config = config

    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        cf_model = self.config.collaborative_model()

        model_path = os.path.join(self.config.collaborative_model_dir, f"{repr(cf_model)}.pkl")

        logger.info(f"loading model from {model_path}")
        with open(model_path, "rb") as f:
            cf_model = pickle.load(f)

        user_story = zip(stories["customer_id"], stories["story_id"])
        stories[repr(cf_model)] = [cf_model.predict(story[0], story[1]) for story in user_story]
        
        return stories[["customer_id", "story_id", repr(cf_model)]]
