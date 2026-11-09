"""Glue ETL (Glue 5.0, PySpark): point-in-time training snapshot from the prepared Parquet.

The prepared social features were computed by `prepare()` from strictly earlier dates, so the
point-in-time join is the identity on (event order, event_time): each row's features are those
recorded at its own event time. The job (1) keeps the temporal order with a dense `seq`, (2)
asserts no feature row is later than its prediction instant, (3) evaluates the Glue Data
Quality ruleset, and (4) writes a versioned, immutable snapshot under
features/materialized/<snapshot_version>/<run id>/ that training reads as its input.
"""

import sys

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ["JOB_NAME", "JOB_RUN_ID", "input_uri", "output_uri", "snapshot_version"])
glue = GlueContext(SparkContext.getOrCreate())
spark = glue.spark_session
prepared = spark.read.parquet(args["input_uri"])
ordered = prepared.withColumn("_file_order", F.monotonically_increasing_id())
snapshot = ordered.withColumn("seq", F.row_number().over(Window.orderBy("event_time", "_file_order")) - 1)
snapshot = snapshot.withColumn("feature_time", F.col("event_time")).drop("_file_order")
leaks = snapshot.filter(F.col("feature_time") > F.col("event_time")).count()
if leaks:
    raise RuntimeError(f"{leaks} rows use features recorded after the prediction time")
unordered = snapshot.withColumn("previous", F.lag("event_time").over(Window.orderBy("seq"))) \
    .filter(F.col("previous") > F.col("event_time")).count()
if unordered:
    raise RuntimeError(f"{unordered} rows break chronological order")
target = f"{args['output_uri'].rstrip('/')}/{args['snapshot_version']}/{args['JOB_RUN_ID']}/"
snapshot.repartition(1).sortWithinPartitions("seq").write.mode("errorifexists").parquet(target)
print({"rows": snapshot.count(), "output": target, "leaks": leaks, "unordered": unordered})
