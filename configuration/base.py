from pipeline.config_base import ConfigBase
from pipeline.beautifier.beautifier import DataBeautifier
from pipeline.feature_extractors.base import FeatureExtractorCombiner
import lightgbm


class Config(ConfigBase):
    def __init__(self):
        data_beautifier = DataBeautifier(self)

        feature_extractor = FeatureExtractorCombiner([])

        model = lambda : lightgbm.LGBMRegressor()

        super().__init__(
            experiment_name="base",
            data_beautifier=data_beautifier,
            feature_extractor=feature_extractor,
            model=model
        )
