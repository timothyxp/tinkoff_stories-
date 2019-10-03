from .base import FeatureExtractorBase
import pandas as pd
from pipeline.config_base import ConfigBase


class FeatureExtractorStaticDescriptions(FeatureExtractorBase):
    def __init__(self, config: ConfigBase):
        self.config = config

    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        description = pd.read_csv(self.config.stories_descriptions_parsed) \
            ["story_id", "icon_alpha_0"]

        candidates = candidates \
            .merge(description, on="story_id", how="left")

        return candidates

