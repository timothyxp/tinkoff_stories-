import logging

from pyspark import SparkContext, SQLContext, SparkConf

from pipeline import DATA_PATH
from typing import Optional


SPARK_CONTEXT: Optional[SparkContext] = None
SQL_CONTEXT: Optional[SQLContext] = None


def config_pyspark_submit_args():
    sc_conf = SparkConf()

    sc_conf.set("spark.driver.memory", "20g")

    sc_conf.set("spark.driver.maxResultSize", "0")

    sc_conf.set("spark.executor.memory", "20g")

    sc_conf.set("spark.shuffle.memoryFraction", "0.3")

    sc_conf.set("spark.python.worker.memory", "8g")

    sc_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    sc_conf.set("spark.memory.fraction", "0.4")

    sc_conf.set("spark.memory.storageFraction", "0.5")

    sc_conf.set("spark.local.dir", f"{DATA_PATH}/spark_tmp")
    sc_conf.set("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={DATA_PATH}/spark_tmp")
    sc_conf.set("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={DATA_PATH}/spark_tmp")

    sc_conf.set("spark.sql.execution.arrow.enabled", "true")

    sc_conf.set("spark.sql.shuffle.partitions", "100")

    return sc_conf


def setup_context():
    global SPARK_CONTEXT
    global SQL_CONTEXT

    config = config_pyspark_submit_args()

    SPARK_CONTEXT = SparkContext(conf=config)
    SQL_CONTEXT = SQLContext(SPARK_CONTEXT)

    logging.getLogger('py4j').setLevel(logging.ERROR)

    SPARK_CONTEXT.setLogLevel("ERROR")

    SPARK_CONTEXT.setCheckpointDir(f"{DATA_PATH}/checkpoint/")


def get_sql_context() -> SQLContext:
    return SQL_CONTEXT

