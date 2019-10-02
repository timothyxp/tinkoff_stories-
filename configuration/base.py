from pipeline.config_base import ConfigBase
from pipeline.beautifier.beautifier import DataBeautifier
from pipeline.feature_extractors.base import FeatureExtractorCombiner
from pipeline.feature_extractors.als import FeatureExtractorALS
from pipeline.feature_extractors.age_features import FeatureExtractorCustomerAge
from pipeline.feature_extractors.gender_features import FeatureExtractorCustomerGender
from pipeline.feature_extractors.childer_features import FeatureExtractorCustomerChildrenAmount
from pipeline.feature_extractors.job_category import FeatureExtractorCustomerJobCategory
import lightgbm
from pipeline.collaborative_models.als import ALS


class Config(ConfigBase):
    def __init__(self):
        data_beautifier = DataBeautifier(self)

        feature_extractor = FeatureExtractorCombiner([
            FeatureExtractorALS(self),
            FeatureExtractorCustomerChildrenAmount(),
            FeatureExtractorCustomerJobCategory(),
            FeatureExtractorCustomerGender(),
            FeatureExtractorCustomerAge()
        ])

        model = lambda : lightgbm.LGBMClassifier(
            class_weight={
                0: 1,
                1: 0.1,
                2: 0.1,
                3: 0.5
            },
            learning_rate=0.1,
            num_leaves=31,
            n_estimators=100
        )

        collaborative_model = lambda : ALS()

        score_mapper = {
            0: -1,
            1: -0.9,
            2: 1,
            3: 1,
        }

        super().__init__(
            experiment_name="base",
            data_beautifier=data_beautifier,
            feature_extractor=feature_extractor,
            model=model,
            collaborative_model=collaborative_model,
            score_mapper=score_mapper
        )
