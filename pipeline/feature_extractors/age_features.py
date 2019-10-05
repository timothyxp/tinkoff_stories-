import pandas as pd

from .base import FeatureExtractorBase


def categorize_age(row):
    if row.customer_age <= 18:
        return 0
    if 18 < row.customer_age <= 26:
        return 1
    if 26 < row.customer_age <= 44:
        return 2
    return 3


class FeatureExtractorAgeCategoryAndAge(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame,
                candidates: pd.DataFrame) -> pd.DataFrame:
        age = users[['age', 'customer_id']].drop_duplicates(subset=['customer_id'])
        age = age.rename(columns={'age': 'customer_age'})
        age['age_category'] = age.apply(lambda row: categorize_age(row), axis=1)
        age.age_category = age.age_category.astype('category')
        candidates = candidates.merge(age, on='customer_id', how='left')
        return candidates
