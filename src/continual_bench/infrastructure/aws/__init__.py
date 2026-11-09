"""AWS adapters (boto3). Imported only by application/orchestration code paths of the AWS profile.

`core`, `models` and `detectors` never import this package; boto3 loads lazily inside each adapter.
"""
