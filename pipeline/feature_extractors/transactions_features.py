import pandas as pd

from .base import FeatureExtractorBase


class FeatureExtractorCustomerSumTransactionAmt(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        user_transactions = transactions[['customer_id', 'transaction_amt']]

        user_transactions_sum = user_transactions.groupby('customer_id').transaction_amt.agg('sum').to_frame()

        candidates = candidates.merge(user_transactions_sum, on='customer_id', how='left')
        candidates = candidates.rename(columns={'transaction_amt': 'customer_sum_amt'})
        return candidates


class FeatureExtractorMinMaxTransactionAmt(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        user_transactions = transactions[['customer_id', 'transaction_amt']]

        user_transactions_minmax = user_transactions.groupby('customer_id').agg(['min', 'max'])
        user_transactions_minmax = user_transactions_minmax.transaction_amt
        user_transactions_minmax = user_transactions_minmax.rename(columns={'min': 'customer_min_amt',
                                                                            'max': 'customer_max_amt'})
        candidates = candidates.merge(user_transactions_minmax, on='customer_id', how='left')
        return candidates


class FeatureExtractorAvgTransactionAmt(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        user_transactions = transactions[['customer_id', 'transaction_amt']]

        user_transactions_mean = user_transactions.groupby('customer_id').transaction_amt.agg('mean').to_frame()

        candidates = candidates.merge(user_transactions_mean, on='customer_id', how='left')
        candidates = candidates.rename(columns={'transaction_amt': 'customer_avg_amt'})
        return candidates