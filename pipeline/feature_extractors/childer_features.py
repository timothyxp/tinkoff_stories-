import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerChildrenAmount(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        children_amount = users[['children_cnt', 'customer_id']]
        children_amount = children_amount.rename(columns={'children_cnt': 'customer_children_cnt'})
        stories = stories.merge(children_amount, on='customer_id', how='left')
        return stories