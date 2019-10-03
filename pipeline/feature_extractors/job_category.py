import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerJobCategory(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        job_category = users[['job_position_cd', 'customer_id']]
        job_category = job_category.rename(columns={'job_position_cd': 'customer_job_position_cd'})
        job_category['customer_job_position_cd'] = job_category['customer_job_position_cd'].astype('category')
        candidates = candidates.merge(job_category, on='customer_id', how='left')
        return candidates


class FeatureExtractorCustomerJobTitleTransactionMean(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        trans_amt_mean = transactions.groupby(['customer_id'])['transaction_amt'].mean().reset_index()
        cust_mean = users.merge(trans_amt_mean, on=['customer_id'], how='left')[
            ['customer_id', 'job_position_cd', 'transaction_amt']]
        cust_mean_job_title = cust_mean.groupby(['job_position_cd'])['transaction_amt'].mean().reset_index()
        cust_mean_job = cust_mean[['customer_id', 'job_position_cd']].merge(cust_mean_job_title, on=['job_position_cd'],
                                                                            how='left')
        cust_mean_job = cust_mean_job.rename(columns={'transaction_amt': 'transaction_amt_job_position'})
        candidates = candidates.merge(cust_mean_job[['customer_id', 'transaction_amt_job_position']],
                                      on=['customer_id'], how='left')
        return candidates


class FeatureExtractorCustomerJobPositionClassify(FeatureExtractorBase):
    def extract(self,  transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        not_working = [9, 11, 13, 14, 15]
        supervisor = [2, 3, 4, 8]
        worker = [1, 7, 10, 16, 17, 19, 20, 21]

        def job_position_classifier(cd):
            if cd in not_working:
                return 'not_working'
            if cd in supervisor:
                return 'supervisor'
            if cd in worker:
                return 'worker'
            return 'unknown'

        cust_test_job_status_cd = users[['customer_id', 'job_position_cd']].copy()
        cust_test_job_status_cd['job_position_cd'] = cust_test_job_status_cd['job_position_cd'].apply(
            job_position_classifier)
        for job_position in cust_test_job_status_cd['job_position_cd'].unique():
            cust_test_job_status_cd['job_position_cd_' + str(job_position)] = cust_test_job_status_cd[
                'job_position_cd'].apply(lambda x: 1 if x == job_position else 0)

        cust_test_job_status_cd = cust_test_job_status_cd.drop(['job_position_cd'], axis=1)
        candidates = candidates.merge(cust_test_job_status_cd, on=['customer_id'], how='left')
        return candidates

