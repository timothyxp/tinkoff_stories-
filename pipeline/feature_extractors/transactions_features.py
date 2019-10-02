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


class FeatureExtractorAvgTransactionAmtByMonth(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        user_transactions = transactions[['customer_id', 'transaction_amt']]
        for i in [5, 6, 7]:
            transactions_mean_bill_month = transactions[transactions['transaction_month'] == i].groupby(['customer_id'])[
                'transaction_amt'].mean().reset_index()
            transactions_mean_bill_month = transactions_mean_bill_month.rename(
                columns={'transaction_amt': 'transaction_amt_mean_month_' + str(i)})
            user_transactions = user_transactions.merge(transactions_mean_bill_month, on=['customer_id'], how='left')
        user_transactions = user_transactions.fillna(0)
        user_transactions = user_transactions.drop(['transaction_amt'], axis=1)
        return user_transactions


class FeatureExtractorAvgMerchantUnique(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        user_transactions = transactions[['customer_id', 'merchant_id', 'merchant_mcc']].copy()
        user_transactions_count_unique_id = user_transactions.groupby(
            ['customer_id'])['merchant_id'].nunique().reset_index()

        user_transactions_count_unique_id = user_transactions_count_unique_id.rename(
            columns={'merchant_id': 'merchant_id_unique'})

        user_transactions_count_unique_mcc  = user_transactions.groupby(
            ['customer_id'])['merchant_mcc'].nunique().reset_index()

        user_transactions_count_unique_mcc = user_transactions_count_unique_mcc.rename(
            columns={'merchant_id': 'merchant_mcc_unique'})

        user_transactions = user_transactions.merge(user_transactions_count_unique_id, on=['customer_id'], how='left')
        user_transactions = user_transactions.merge(user_transactions_count_unique_mcc, on=['customer_id'], how='left')
        return user_transactions[['customer_id', 'merchant_id_unique', 'merchant_mcc_unique']]


class FeatureExtractorAvgMeanTransactionAmtOnMerchant(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        transactions_merchant = transactions.groupby(['merchant_id'])['transaction_amt'].mean().reset_index()
        transactions_merchant = transactions_merchant.rename(columns={'transaction_amt': 'transaction_amt_by_merchant'})
        transactions_merchant_mean = transactions.copy()
        transactions_merchant_mean = transactions_merchant_mean.merge(transactions_merchant, on=['merchant_id'],
                                                                      how='left')
        transactions_merchant_mean_customer_id = transactions_merchant_mean.groupby(['customer_id'])[
            'transaction_amt_by_merchant'].mean().reset_index()

        user_transactions = transactions.copy()
        user_transactions = user_transactions.merge(transactions_merchant_mean_customer_id, on=['customer_id'],
                                                    how='left')

        return user_transactions[['customer_id', 'transaction_amt_by_merchant']]







