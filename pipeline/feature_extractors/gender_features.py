import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerAge(FeatureExtractorBase):
    def __init__(self, config):
        self.config = config

    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        gender = users[['gender_rk', 'customer_id']]
        gender = gender.rename(columns={'gender_cd': 'customer_gender'})
        gender['customer_gender'] = [0 if gen == 'M' else 1 for gen in gender['customer_gender']]
        stories = stories.merge(gender, on='customer_id', how='left')
        return stories