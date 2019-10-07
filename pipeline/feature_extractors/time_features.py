import pandas as pd
from datetime import datetime
import time

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


class FeatureExtractorTransactionInWeekDay(FeatureExtractorBase):
    def week_day(self, df):
        dt = datetime.datetime(year=2018, month=df['transaction_month'], day=df['transaction_day'])
        time.mktime(dt.timetuple())
        return dt.weekday()

    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        transactions['event_wday'] = transactions.progress_apply(self.week_day, axis=1)
        transactions_in_day = transactions.groupby(['event_dt'])['transaction_amt'].agg(['max',
                                                                                         'mean',
                                                                                         'median']).reset_index()
        candidates['event_dttm'] = pd.to_datetime(candidates['event_dttm'])
        candidates['event_wday'] = candidates['event_dttm'].apply(lambda x: x.weekday())
        candidates = candidates.merge(transactions_in_day, on=['event_wday'], how='left')
        return candidates


class FeatureExtractorTransactionInWeek(FeatureExtractorBase):
    def week(self, df):
        dt = datetime.datetime(year=2018, month=df['transaction_month'], day=df['transaction_day'])
        time.mktime(dt.timetuple())
        return dt.isocalendar()[1]

    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        transactions['event_dt'] = transactions.progress_apply(self.week_day, axis=1)
        transactions['event_week'] = transactions.progress_apply(self.week, axis=1)
        transactions_in_week = transactions.groupby(['event_week'])['transaction_amt'].agg(['max',
                                                                                            'mean',
                                                                                            'median']).reset_index()
        candidates['event_dttm'] = pd.to_datetime(candidates['event_dttm'])
        candidates['event_week'] = candidates['event_dttm'].apply(lambda x: x.weekday())
        candidates = candidates.merge(transactions_in_week, on=['event_week'], how='left')
        candidates
        return candidates


class FeatureExtractorLikeInWeekDay(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        stories['event_dttm'] = pd.to_datetime(stories['event_dttm'])
        stories['event_wday'] = stories['event_dttm'].apply(lambda x: x.weekday())
        events_list = ['like', 'view', 'skip', 'dislike']
        stories_in_day = stories.drop_duplicates(['event_wday']).reset_index(drop=True)
        for event in events_list:
            stories_in_day_x = stories[stories['event'] == event].groupby(
                ['event_wday']).apply(len).reset_index()
            stories_in_day = stories_in_day.merge(stories_in_day_x, on=['event_wday'], how='left')
            stories_in_day = stories_in_day.rename(columns={stories_in_day_x.columns[1]: event})
        candidates['event_dttm'] = pd.to_datetime(candidates['event_dttm'])
        candidates['event_wday'] = candidates['event_dttm'].apply(lambda x: x.weekday())
        candidates = candidates.merge(stories_in_day, on=['event_wday'], how='left')
        return candidates[['story_id', 'like', 'view', 'skip', 'dislike']]


class FeatureExtractorLikeInWeek(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        stories['event_dttm'] = pd.to_datetime(stories['event_dttm'])
        stories['event_week'] = stories['event_dttm'].apply(lambda x: x.isocalendar()[1])
        events_list = ['like', 'view', 'skip', 'dislike']
        stories_in_week = stories.drop_duplicates(['event_week']).reset_index(drop=True)
        for event in events_list:
            stories_in_week_x = stories[stories['event'] == event].groupby(
                ['event_week']).apply(len).reset_index()
            stories_in_week = stories_in_week.merge(stories_in_week_x, on=['event_wday'], how='left')
            stories_in_week = stories_in_week.rename(columns={stories_in_week_x.columns[1]: event})
        candidates['event_dttm'] = pd.to_datetime(candidates['event_dttm'])
        candidates['event_week'] = candidates['event_dttm'].apply(lambda x: x.isocalendar()[1])
        candidates = candidates.merge(stories_in_week, on=['event_week'], how='left')
        return candidates[['story_id', 'like', 'view', 'skip', 'dislike']]