import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerAge(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        age = users[['age', 'customer_id']].drop_duplicates(subset=["customer_id"])
        age = age.rename(columns={'age': 'customer_age'})
        candidates = candidates.merge(age, on='customer_id', how='left')
        return candidates
