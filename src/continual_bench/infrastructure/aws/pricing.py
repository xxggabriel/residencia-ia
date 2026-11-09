"""AWS Price List API → versioned regional estimate file. Never called during training."""

import json
from datetime import date

from .session import client as make_client

LOCATION = {"us-east-1": "US East (N. Virginia)"}

# (service code, filters, label, unit) for the resources the campaign reserves or shares.
PRODUCTS = {
    "fargate_vcpu_hour": ("AmazonECS", {"usagetype": "USE1-Fargate-vCPU-Hours:perCPU"}),
    "fargate_memory_gb_hour": ("AmazonECS", {"usagetype": "USE1-Fargate-GB-Hours"}),
    "fargate_ephemeral_storage_gb_hour": (
        "AmazonECS",
        {"usagetype": "USE1-Fargate-EphemeralStorage-GB-Hours"},
    ),
    "s3_standard_gb_month": ("AmazonS3", {"usagetype": "TimedStorage-ByteHrs", "volumeType": "Standard"}),
    "vpc_interface_endpoint_hour": ("AmazonVPC", {"usagetype": "USE1-VpcEndpoint-Hours"}),
    "vpc_interface_endpoint_gb": ("AmazonVPC", {"usagetype": "USE1-VpcEndpoint-Bytes"}),
    "cloudwatch_logs_ingest_gb": ("AmazonCloudWatch", {"usagetype": "USE1-DataProcessing-Bytes"}),
    "kinesis_shard_hour": ("AmazonKinesis", {"usagetype": "USE1-Storage-ShardHour"}),
    "codebuild_general1_medium_minute": ("CodeBuild", {"usagetype": "USE1-Build-Min:g1.medium"}),
    "lambda_gb_second": ("AWSLambda", {"usagetype": "USE1-Lambda-GB-Second"}),
}


def first_tier_usd(product: dict) -> tuple[float, str, str]:
    terms = product["terms"]["OnDemand"]
    offer = next(iter(terms.values()))
    dimensions = sorted(offer["priceDimensions"].values(), key=lambda d: float(d.get("beginRange", 0)))
    first = dimensions[0]
    return float(first["pricePerUnit"]["USD"]), first["unit"], offer["effectiveDate"]


def fetch(region="us-east-1", *, client=None) -> dict:
    client = client or make_client("pricing", region_name="us-east-1")
    prices = {}
    for name, (service, filters) in PRODUCTS.items():
        terms = [{"Type": "TERM_MATCH", "Field": "location", "Value": LOCATION[region]}]
        terms += [{"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in filters.items()]
        products = client.get_products(ServiceCode=service, Filters=terms, MaxResults=10)["PriceList"]
        if len(products) != 1:
            prices[name] = {"status": "ambiguous_or_missing", "matches": len(products), "filters": filters}
            continue
        product = json.loads(products[0])
        usd, unit, effective = first_tier_usd(product)
        prices[name] = {
            "usd": usd,
            "unit": unit,
            "effective_date": effective,
            "service": service,
            "sku": product["product"]["sku"],
            "filters": filters,
            "status": "verified_price_list_api",
        }
    return {
        "version": "aws-price-snapshot-v1",
        "region": region,
        "retrieved": date.today().isoformat(),
        "source": "AWS Price List Query API (pricing.us-east-1.amazonaws.com)",
        "prices": prices,
    }
