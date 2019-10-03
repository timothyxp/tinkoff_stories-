import pandas as pd
from datetime import datetime

from .base import FeatureExtractorBase


class FeatureExtractorHourCategory(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        candidates['event_datetime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in candidates.event_dttm]
        candidates['hour_category'] = [x.hour for x in candidates.event_datetime]
        candidates.drop(columns=['event_datetime'], inplace=True)
        candidates.hour_category = candidates.hour_category.astype('category')
        return candidates


class FeatureExtractorDayCategory(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        candidates['event_datetime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in candidates.event_dttm]
        candidates['week_day'] = [x.weekday() for x in candidates.event_datetime]
        candidates.drop(columns=['event_datetime'], inplace=True)
        candidates.week_day = candidates.week_day.astype('category')
        return candidates

