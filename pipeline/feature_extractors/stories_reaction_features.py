import pandas as pd
from datetime import datetime

from .base import FeatureExtractorBase


class FeatureExtractorMeanLikeValueForCustomer(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        ans_to_int = {'dislike': 0, 'skip' : 1, 'view' : 2, 'like': 3}
        stories_event_int = stories.copy()
        stories_event_int['event'] = stories_event_int['event'].map(ans_to_int)
        stories_event_int = stories_event_int.groupby('customer_id').event.agg(['sum', 'mean']).reset_index()
        return candidates.merge(stories_event_int, how='left', on='customer_id')


class FeatureExtractorMeanLikeValueForStory(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        ans_to_int = {'dislike': 0, 'skip' : 1, 'view' : 2, 'like': 3}
        stories_event_int = stories.copy()
        stories_event_int['event'] = stories_event_int['event'].map(ans_to_int)
        stories_event_int = stories_event_int.groupby('story_id').event.agg(['sum', 'mean']).reset_index()
        return candidates.merge(stories_event_int, how='left', on='story_id')

    