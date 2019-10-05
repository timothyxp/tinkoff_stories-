from pipeline.config_base import ConfigBase
from pipeline.beautifier.beautifier import DataBeautifier
from pipeline.feature_extractors.base import FeatureExtractorCombiner
from pipeline.feature_extractors.als import FeatureExtractorALS
from pipeline.feature_extractors.age_features import FeatureExtractorAgeCategoryAndAge
from pipeline.feature_extractors.gender_features import FeatureExtractorCustomerGender
from pipeline.feature_extractors.childer_features import FeatureExtractorCustomerChildrenAmount
from pipeline.feature_extractors.transactions_features import FeatureExtractorAvgTransactionAmt, \
    FeatureExtractorCustomerSumTransactionAmt, FeatureExtractorMinMaxTransactionAmt, \
    FeatureExtractorAvgMeanTransactionAmtOnMerchant, FeatureExtractorAvgMerchantUnique, \
    FeatureExtractorAvgTransactionAmtByMonth
from pipeline.feature_extractors.stories_reaction_features import FeatureExtractorMeanLikeValueForCustomer,  \
    FeatureExtractorMeanLikeValueForStory, FeatureExtractorStoriesReactionsAmount, \
    FeatureExractorUserStoriesReactionsAmount, FeatureExtractorDuplicatedReaction
from pipeline.feature_extractors.time_features import FeatureExtractorDayCategory, FeatureExtractorHourCategory
from pipeline.feature_extractors.marital_features import FeatureExtractorCustomerMaritalCategories
from pipeline.feature_extractors.job_category import FeatureExtractorCustomerJobCategory, \
    FeatureExtractorCustomerJobPositionClassify, FeatureExtractorCustomerJobTitleTransactionMean
from pipeline.feature_extractors.descriptions import FeatureExtractorStaticDescriptions, \
    FeatureExtractorDescriptionsFromModel, FeatureExtractorDescriptionsFelix
import lightgbm
from pipeline.collaborative_models.als import ALS
from catboost import CatBoostClassifier


class Config(ConfigBase):
    def __init__(self):
        data_beautifier = DataBeautifier(self)

        feature_extractor = FeatureExtractorCombiner([
            FeatureExtractorALS(self),
            FeatureExtractorCustomerChildrenAmount(),
            FeatureExtractorCustomerJobCategory(),
            FeatureExtractorCustomerGender(),
            FeatureExtractorAgeCategoryAndAge(),
            FeatureExtractorMinMaxTransactionAmt(),
            FeatureExtractorCustomerSumTransactionAmt(),
            FeatureExtractorAvgTransactionAmt(),
            FeatureExtractorAvgMeanTransactionAmtOnMerchant("merchant_id"),
            FeatureExtractorAvgMeanTransactionAmtOnMerchant("merchant_mcc"),
            FeatureExtractorAvgMerchantUnique(),
            FeatureExtractorAvgTransactionAmtByMonth(),
            FeatureExtractorHourCategory(),
            FeatureExtractorDayCategory(),
            FeatureExtractorMeanLikeValueForStory(),
            FeatureExtractorMeanLikeValueForCustomer(),
            FeatureExtractorStoriesReactionsAmount(),
            FeatureExractorUserStoriesReactionsAmount(),
            FeatureExtractorCustomerMaritalCategories(),
            FeatureExtractorCustomerJobPositionClassify(),
            FeatureExtractorCustomerJobTitleTransactionMean(),
            FeatureExtractorDescriptionsFromModel('stories_desc.csv'),
            FeatureExtractorDuplicatedReaction(),
            FeatureExtractorStaticDescriptions(self),
            FeatureExtractorDescriptionsFelix("felix_descriptions.csv")
        ])

        model = lambda : CatBoostClassifier(
            learning_rate=0.07,
            max_depth=2,
            iterations=70,
            thread_count=8
        )

        collaborative_model = lambda : ALS()

        score_mapper = {
            0: -1,
            1: -1,
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
