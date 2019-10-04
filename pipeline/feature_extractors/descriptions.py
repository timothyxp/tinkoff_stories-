from .base import FeatureExtractorBase
import pandas as pd
from pipeline.config_base import ConfigBase
from colour import Color
import numpy as np


class FeatureExtractorStaticDescriptions(FeatureExtractorBase):
    def __init__(self, config: ConfigBase):
        self.config = config

    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        description = pd.read_csv(self.config.stories_descriptions_parsed) \
            [["story_id", "icon_alpha_0", "icon_hex_0", "icon_hex_1"]]

        candidates = candidates \
            .merge(description, on="story_id", how="left")

        candidates["icon_0_red"] = [Color(hex).rgb[0] if hex is str else 0 for hex in candidates["icon_hex_0"]]
        candidates["icon_0_green"] = [Color(hex).rgb[1] if hex is str else 0 for hex in candidates["icon_hex_0"]]
        candidates["icon_0_blue"] = [Color(hex).rgb[2] if hex is str else 0 for hex in candidates["icon_hex_0"]]
        candidates["icon_0_red"] = [Color(hex).rgb[0] if hex is str else 0 for hex in candidates["icon_hex_1"]]
        candidates["icon_0_green"] = [Color(hex).rgb[1] if hex is str else 0 for hex in candidates["icon_hex_1"]]
        candidates["icon_0_blue"] = [Color(hex).rgb[2] if hex is str else 0 for hex in candidates["icon_hex_1"]]

        candidates.drop(columns=["icon_hex_0", "icon_hex_1"], inplace=True)

        return candidates

