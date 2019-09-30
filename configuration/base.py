from pipeline.config_base import ConfigBase
from pipeline.beautifier.beautifier import DataBeautifier
from pipeline.feature_extractors.base import FeatureExtractorCombiner
from pipeline.feature_extractors.als import FeatureExtractorALS
import lightgbm
from pipeline.collaborative_models.als import ALS


class Config(ConfigBase):
    def __init__(self):
        data_beautifier = DataBeautifier(self)

        feature_extractor = FeatureExtractorCombiner([
            FeatureExtractorALS(self)
        ])

        model = lambda : lightgbm.LGBMRegressor()

        collaborative_model = lambda : ALS()

        super().__init__(
            experiment_name="base",
            data_beautifier=data_beautifier,
            feature_extractor=feature_extractor,
            model=model,
            collaborative_model=collaborative_model
        )
