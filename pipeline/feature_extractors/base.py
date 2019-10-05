import abc
from pipeline.logging.logger import logger
import pandas as pd
from typing import List


class FeatureExtractorBase(abc.ABC):
    @abc.abstractmethod
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        pass

    def __repr__(self):
        return self.__class__.__name__


class FeatureExtractorCombiner(FeatureExtractorBase):
    def __init__(self,
                 feature_extractors: List[FeatureExtractorBase],
                 add_extractor_prefix_name: bool=False
        ):
        self.add_extractor_prefix_name = add_extractor_prefix_name
        self._feature_extractors = feature_extractors


    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        logger.info("start extract features from combiner")

        result = candidates.copy(deep=True)
        logger.debug(f"candidates columns {candidates.columns}")

        merge_columns = ["customer_id", "story_id", "event_dttm"]

        candidates_columns_len = len(merge_columns)

        candidates = candidates[merge_columns]

        for feature_extractor in self._feature_extractors:
            logger.info(f"start extract from {repr(feature_extractor)}")

            features = feature_extractor.extract(transactions.copy(), stories.copy(), users.copy(), candidates.copy())
            features = features.drop_duplicates(subset=merge_columns)
            features_count = len(features.columns) - candidates_columns_len

            logger.debug(f"get {features_count} features")
            logger.debug(f"feature columns = {features.columns}")

            if features_count == 0:
                logger.warning(f"{repr(feature_extractor)} doesnt return features")

            logger.debug(f"shape before merge {result.shape}")
            result = result.merge(features, on=merge_columns, how="left")
            logger.debug(f"shape after merge {result.shape}")

        return result.drop_duplicates(
            subset=merge_columns
        )

    def __repr__(self):
        reprs = [repr(feature_extractor) for feature_extractor in self._feature_extractors]
        return "combiner_{}_".format("_".join(reprs))
