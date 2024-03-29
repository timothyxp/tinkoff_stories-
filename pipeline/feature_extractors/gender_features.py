import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerGender(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        gender = users[['gender_cd', 'customer_id']]
        gender = gender.rename(columns={'gender_cd': 'customer_gender'})
        gender['customer_gender'] = [0 if gen == 'M' else 1 for gen in gender['customer_gender']]
        gender['customer_gender'] = gender['customer_gender'].astype('category')
        candidates = candidates.merge(gender, on='customer_id', how='left')
        return candidates