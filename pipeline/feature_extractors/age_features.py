import pandas as pd

from .base import FeatureExtractorBase


def categorize_age(row):
    if row.age <= 18:
        return 0
    if 18 < row.age <= 26:
        return 1
    if 26 < row.age <= 44:
        return 2
    return 3


class FeatureExtractorAgeCategoryAndAge(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame,
                candidates: pd.DataFrame) -> pd.DataFrame:
        age = users[['age', 'customer_id']].drop_duplicates(subset=['customer_id'])
        age['age_category'] = age.apply(lambda row: categorize_age(row), axis=1)
        age = age.rename(columns={'age': 'customer_age'})
        age.age_category = age.age_category.astype('category')
        candidates = candidates.merge(age, on='customer_id', how='left')
        return candidates


class FeatureExtractorCountForEachCategory(FeatureExtractorBase):
    def extract(self, transactions: pd.DataFrame, stories: pd.DataFrame, users: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        age = users[['age', 'customer_id']].drop_duplicates(subset=['customer_id'])
        age['age_category'] = age.apply(lambda row: categorize_age(row), axis=1)
        age = age.rename(columns={'age': 'customer_age'})

        candidates = candidates.merge(users[['customer_id', 'age']])
        candidates['age_category'] = candidates.apply(lambda row: categorize_age(row), axis=1)
        stories = stories.merge(age, on='customer_id', how='left')
        ans_to_int = {'dislike': 0, 'skip': 1, 'view': 2, 'like': 3}
        stories['event'] = stories['event'] \
            .map(ans_to_int)

        for i in range(4):
            stories_tmp = stories[stories.event == i]
            stories_tmp = stories_tmp.groupby('age_category').agg('count').reset_index()
            stories_tmp = stories_tmp[['age_category', 'customer_id']]
            stories_tmp = stories_tmp.rename(columns={'customer_id': f'customer_count_on_{i}'})
            candidates = candidates.merge(stories_tmp, on='age_category', how='left')

        for i in range(4):
            candidates[f'customer_percentage_on_{i}'] = candidates[f'customer_count_on_{i}'] / \
                (candidates[f'customer_count_on_0'] + candidates[f'customer_count_on_1'] + candidates[f'customer_count_on_2'] + candidates[f'customer_count_on_3'])

        candidates.drop(columns=['age_category'], inplace=True)
        candidates.drop(columns=['age'], inplace=True)
        return candidates
