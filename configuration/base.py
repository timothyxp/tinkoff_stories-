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

        model = lambda : lightgbm.LGBMClassifier(
            class_weight={
                0: 10,
                1: 100,
                2: 100,
                3: 20
            },
            learning_rate=0.1,
            num_leaves=31,
            n_estimators=100
        )

        collaborative_model = lambda : ALS()

        super().__init__(
            experiment_name="base",
            data_beautifier=data_beautifier,
            feature_extractor=feature_extractor,
            model=model,
            collaborative_model=collaborative_model
        )
