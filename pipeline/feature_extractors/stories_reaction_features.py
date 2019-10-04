import pandas as pd
from datetime import datetime

from .base import FeatureExtractorBase


class FeatureExtractorMeanLikeValueForCustomer(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        ans_to_int = {'dislike': 0, 'skip' : 1, 'view' : 2, 'like': 3}

        stories_event_int = stories.copy()

        stories_event_int['event'] = stories_event_int['event'] \
            .map(ans_to_int)

        stories_event_int = stories_event_int \
            .groupby('customer_id') \
            .event \
            .agg(['sum', 'mean', 'min', 'max', 'std']) \
            .reset_index()

        stories_event_int = stories_event_int \
            .rename(columns={
            'sum' : 'customer_sum_of_like_value',
            'mean': 'customer_mean_of_like_value',
            'min': 'customer_min_of_like_value',
            'max': 'customer_max_of_like_value',
            'std': 'customer_std_of_like_value'
        })

        return candidates \
            .merge(stories_event_int, how='left', on='customer_id')


class FeatureExtractorMeanLikeValueForStory(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        ans_to_int = {'dislike': 0, 'skip' : 1, 'view' : 2, 'like': 3}

        stories_event_int = stories.copy()

        stories_event_int['event'] = stories_event_int['event'] \
            .map(ans_to_int)

        stories_event_int = stories_event_int \
            .groupby('story_id') \
            .event \
            .agg(['sum', 'mean', 'min', 'max', 'std']) \
            .reset_index()

        stories_event_int = stories_event_int \
            .rename(columns={
            'sum': 'story_sum_of_like_value',
            'mean': 'story_mean_of_like_value',
            'min': 'story_min_of_like_value',
            'max': 'story_max_of_like_value',
            'std': 'story_std_of_like_value'
        })

        return candidates.merge(stories_event_int, how='left', on='story_id')


class FeatureExractorOntHotEncodingStories(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        stories_reaction: pd.DataFrame = stories \
            [["customer_id", "event"]] \
            .groupby("customer_id") \
            .event.agg({
            "dislike_amount": lambda x: len(list(filter(lambda y: y == "dislike", x))),
            "skip_amount": lambda x: len(list(filter(lambda y: y == "skip", x))),
            "view_amount": lambda x: len(list(filter(lambda y: y == "view", x))),
            "like_amount": lambda x: len(list(filter(lambda y: y == "like", x)))
        })

        print(stories_reaction.columns)

        candidates = candidates \
            .merge(stories_reaction, on="customer_id", how="left")

        return candidates
