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
