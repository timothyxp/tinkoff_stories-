import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerJobCategory(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        job_category = users[['job_position_cd', 'customer_id']]
        job_category = job_category.rename(columns={'job_position_cd': 'customer_job_position_cd'})
        candidates = candidates.merge(job_category, on='customer_id', how='left')
        return candidates