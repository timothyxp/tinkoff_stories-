import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerMaritalCategories(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        cust_test_marital_status_cd = users[['customer_id', 'marital_status_cd']].copy()
        cust_test_marital_status_cd['marital_status_cd'] = cust_test_marital_status_cd['marital_status_cd'].astype(str)
        for marital_status in cust_test_marital_status_cd['marital_status_cd'].unique():
            cust_test_marital_status_cd['marital_status' + str(marital_status)] = cust_test_marital_status_cd[
                'marital_status_cd'].apply(lambda x: 1 if x == marital_status else 0)
        cust_test_marital_status_cd = cust_test_marital_status_cd.drop(['marital_status_cd'], axis=1)
        candidates = candidates.merge(cust_test_marital_status_cd, on=['customer_id'], how='left')
        return candidates
