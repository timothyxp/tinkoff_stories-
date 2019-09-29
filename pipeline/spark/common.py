from functools import reduce
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from .context import get_sql_context


def concatenate(dfs: List[DataFrame]) -> DataFrame:
    return reduce(DataFrame.union, dfs)


def read_csv(path: str) -> DataFrame:
    return get_sql_context() \
        .read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .load(path)


def read_csv_with_types(path: str, str_columns=()) -> DataFrame:
    data = get_sql_context().read.csv(
        path,
        inferSchema=True,
        header=True
    )

    for column in str_columns:
        if column in data.columns:
            data = data.withColumn(column, data[column].cast(StringType()))

    return data


def write_csv(df: DataFrame, path: str, single=False) -> None:
    if single:
        df = df.coalesce(1)

    df \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .format("com.databricks.spark.csv") \
        .save(path, index="False")


def read_parquet(path: str) -> DataFrame:
    return get_sql_context().read.parquet(path)


