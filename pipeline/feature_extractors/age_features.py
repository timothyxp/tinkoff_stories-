import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerAge(FeatureExtractorBase):
    def __init__(self, config):
        self.config = config

    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        age = users[['age', 'customer_id']]
        age = age.rename(columns={'age': 'customer_age'})
        stories = stories.merge(age, on='customer_id', how='left')
        return stories
