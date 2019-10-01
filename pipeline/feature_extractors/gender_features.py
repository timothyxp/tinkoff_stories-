import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerAge(FeatureExtractorBase):
    def __init__(self, config):
        self.config = config

    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        gender = users[['gender_rk', 'customer_id']]
        gender = gender.rename({'gender_rk': 'customer_gender'})
        stories = stories.merge(gender, on='customer_id', how='left')
        return stories