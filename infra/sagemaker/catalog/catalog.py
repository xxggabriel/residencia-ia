"""Publish immutable benchmark snapshots and associate their finished Batch evaluations."""

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from functools import cached_property

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

DATASETS = {"social": "Social", "fraud": "Fraud", "ecommerce": "E-commerce"}
COMPONENT = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
SNAPSHOT = re.compile(rf"(social|fraud|ecommerce)/({COMPONENT})/({COMPONENT})/(full|smoke)")
MARKER = re.compile(
    rf"datasets/(social|fraud|ecommerce)/({COMPONENT})/manifests/({COMPONENT})/(full|smoke)/_SUCCESS\.json"
)
FINAL = re.compile(rf"campaigns/({COMPONENT})/runs/({COMPONENT})/final\.json")
SHA256 = re.compile(r"[a-f0-9]{64}")
TYPE = "ContinualBenchTabularSnapshot"
FORM = "ContinualBenchTabularSnapshotForm"
COMMON = "amazon.datazone.AssetCommonDetailsFormType"
RUN_HEADING = "## Avaliações recentes no MLflow"
RUN_LINK = re.compile(r"^- \[.*\]\(https://[^\n]+/#/experiments/\d+/runs/[a-f0-9]{32}\?workspace=default\)$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def token(*parts):
    return digest(json.dumps(parts, sort_keys=True).encode())


def metadata_forms(asset):
    return {form["formName"]: json.loads(form["content"]) for form in asset["formsOutput"]}


def readme_with_run(readme, link):
    # Preserve the original guide, including the manually curated legacy evaluation section.
    original, _, recent = readme.partition("\n\n" + RUN_HEADING + "\n")
    links = [line for line in recent.splitlines() if RUN_LINK.fullmatch(line)]
    links = [link, *(line for line in links if line != link)][:20]
    intro = ("\n\n" + RUN_HEADING + "\n\n"
             "Para encontrar todas as avaliações no MLflow, filtre pela tag `cb-catalog-asset-id`.\n\n")
    while links and len(original + intro + "\n".join(links)) > 10240:
        links.pop()
    if not links:
        raise ValueError("No room for the new evaluation link in the catalog README")
    return original + intro + "\n".join(links)


class Publisher:
    def __init__(self, session, *, bucket, domain, project, portal, mlflow_arn, mlflow_ui):
        self.session = session
        self.bucket = bucket
        self.domain = domain
        self.project = project
        self.portal = portal.rstrip("/")
        self.mlflow_arn = mlflow_arn
        self.mlflow_ui = mlflow_ui.rstrip("/")
        self.s3 = session.client("s3")
        self.datazone = session.client("datazone")

    @cached_property
    def form_types(self):
        response = self.datazone.get_asset_type(domainIdentifier=self.domain, identifier=TYPE)
        return response["revision"], response["formsOutput"]

    def read(self, key, version=None):
        version_arg = {"VersionId": version} if version else {}
        with self.s3.get_object(Bucket=self.bucket, Key=key, **version_arg)["Body"] as body:
            return body.read()

    def snapshot(self, parts, version=None):
        dataset, revision, preparation, scope = parts
        snapshot_id = "/".join(parts)
        root = f"datasets/{dataset}/{revision}"
        prefix = f"{root}/manifests/{preparation}/{scope}"
        key = f"{prefix}/snapshot.json"
        marker = json.loads(self.read(f"{prefix}/_SUCCESS.json", version))
        if marker["snapshot_id"] != snapshot_id or marker["manifest_key"] != key:
            raise ValueError("Completion marker does not match its S3 key")
        raw = self.read(key, marker.get("manifest_version_id"))
        if digest(raw) != marker["manifest_sha256"]:
            raise ValueError("Snapshot manifest hash differs from its completion marker")
        manifest = json.loads(raw)
        prepared = manifest["prepared"]
        parquet = prepared["parquet"]
        parquet_key = f"{root}/prepared/{preparation}/{scope}/data.parquet"
        if (manifest["snapshot_id"] != snapshot_id or manifest["version"] != "snapshot-v1"
                or parquet["key"] != parquet_key or parquet["sha256"] != prepared["sha256"]
                or prepared["sha256"] != marker["prepared_sha256"]
                or not SHA256.fullmatch(prepared["sha256"])):
            raise ValueError("Snapshot manifest does not match its prepared object")
        head = self.s3.head_object(Bucket=self.bucket, Key=parquet_key, VersionId=parquet["version_id"])
        if head["Metadata"].get("sha256") != prepared["sha256"] or head["ContentLength"] != prepared["bytes"]:
            raise ValueError("Versioned Parquet metadata differs from its manifest")
        return {
            "SnapshotId": snapshot_id, "ParquetS3Uri": f"s3://{self.bucket}/{parquet_key}",
            "ParquetVersionId": parquet["version_id"], "PreparedSha256": prepared["sha256"],
            "SnapshotManifestS3Uri": f"s3://{self.bucket}/{key}", "SnapshotManifestSha256": digest(raw),
            "SourceRevision": revision, "PreparationVersion": preparation, "DatasetScope": scope,
            "RowCount": manifest["rows"], "License": manifest["license"], "MlflowEvaluationCount": 0,
        }

    def forms(self, values):
        _, types = self.form_types
        return [
            {"formName": name, "typeIdentifier": types[name]["typeName"],
             "typeRevision": types[name]["typeRevision"], "content": json.dumps(content, sort_keys=True)}
            for name, content in values.items()
        ]

    def find_asset(self, fields):
        pages = self.datazone.get_paginator("search").paginate(
            domainIdentifier=self.domain, owningProjectIdentifier=self.project, searchScope="ASSET",
        )
        for page in pages:
            for result in page["items"]:
                item = result.get("assetItem", {})
                if item.get("externalIdentifier") == fields["SnapshotId"]:
                    return self.datazone.get_asset(
                        domainIdentifier=self.domain, identifier=item["identifier"],
                    )
        return None

    def ensure_asset(self, fields, dataset):
        asset = self.find_asset(fields)
        if asset is not None:
            if asset["typeIdentifier"] != TYPE:
                raise ValueError("Snapshot identifier belongs to an unexpected catalog type")
            existing = metadata_forms(asset)["SnapshotMetadata"]
            if any(existing.get(k) != v for k, v in fields.items() if k != "MlflowEvaluationCount"):
                raise ValueError("Catalog entry conflicts with the immutable snapshot")
            return asset
        revision = fields["SourceRevision"]
        scope = fields["DatasetScope"]
        prep = fields["PreparationVersion"]
        type_revision, _ = self.form_types
        return self.datazone.create_asset(
            domainIdentifier=self.domain, owningProjectIdentifier=self.project, typeIdentifier=TYPE,
            typeRevision=type_revision, externalIdentifier=fields["SnapshotId"],
            clientToken=token("asset", self.domain, self.project, fields["SnapshotId"]),
            name=f"Continual Bench — {DATASETS[dataset]} ({scope}, {revision[:12]}, {prep})",
            description="Snapshot Parquet versionado do benchmark tabular Continual Bench.",
            formsInput=self.forms({
                "SnapshotMetadata": fields,
                "AssetCommonDetailsForm": {
                    "sourceIdentifier": fields["ParquetS3Uri"],
                    "readMe": "Snapshot do benchmark tabular. As avaliações são vinculadas após a execução.",
                },
            }),
        )

    def publish(self, asset):
        listing_id = (asset.get("listing") or {}).get("listingId")
        if listing_id:
            current = self.datazone.get_listing(domainIdentifier=self.domain, identifier=listing_id)
            if (current["status"] == "ACTIVE"
                    and current["item"]["assetListing"]["assetRevision"] == asset["revision"]):
                return listing_id
        result = self.datazone.create_listing_change_set(
            domainIdentifier=self.domain, entityIdentifier=asset["id"], entityType="ASSET",
            entityRevision=asset["revision"], action="PUBLISH",
            clientToken=token("publish", asset["id"], asset["revision"]),
        )
        return result["listingId"]

    def mlflow(self, path, *, run_id, tags=None):
        region = self.mlflow_arn.split(":")[3]
        endpoint = f"https://api.mlflow.sagemaker.{region}.app.aws/api/2.0/mlflow/runs/"
        body = json.dumps({"run_id": run_id, "tags": tags}).encode() if tags is not None else b""
        url = endpoint + path if tags is not None else endpoint + path + f"?run_id={run_id}"
        request = AWSRequest(
            method="POST" if tags is not None else "GET", url=url, data=body,
            headers={"x-sm-mlflow-app-arn": self.mlflow_arn, "Accept": "application/json",
                     "Content-Type": "application/json", "X-Amz-Content-SHA256": digest(body)},
        )
        credentials = self.session.get_credentials().get_frozen_credentials()
        SigV4Auth(credentials, "sagemaker", region).add_auth(request)
        signed = request.prepare()
        http_request = urllib.request.Request(
            signed.url, data=body if tags is not None else None,
            headers=dict(signed.headers), method=request.method,
        )
        try:
            with urllib.request.urlopen(http_request, timeout=12) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read(2048))
            except (ValueError, OSError):
                detail = {}
            code = detail.get("error_code", "unknown")
            message = str(detail.get("message", ""))[:300]
            raise RuntimeError(f"MLflow {path} returned HTTP {error.code} ({code}): {message}") from error

    def evaluation(self, event_key, match, version):
        campaign, unit = match.groups()
        final = json.loads(self.read(event_key, version))
        source = final.get("snapshot") or {}
        parts = SNAPSHOT.fullmatch(source.get("id", ""))
        if parts is None or final.get("kind") != "zia":
            return {"status": "outside_catalog_scope"}
        links = final.get("links") or {}
        run_id = links.get("mlflow_run_id")
        if not run_id:
            return {"status": "no_mlflow_run"}
        experiment = str(links["mlflow_experiment_id"])
        if not re.fullmatch(r"[a-f0-9]{32}", run_id) or not experiment.isdecimal():
            raise ValueError("Invalid MLflow identifiers")
        if final["campaign_id"] != campaign or final["run_id"] != unit:
            raise ValueError("Evaluation identity does not match its S3 key")
        dataset, _, _, scope = parts.groups()
        fields = self.snapshot(parts.groups())
        if (source["manifest_sha256"] != fields["SnapshotManifestSha256"]
                or source["prepared_sha256"] != fields["PreparedSha256"]):
            raise ValueError("Evaluation used a different snapshot")
        run = self.mlflow("get", run_id=run_id)["run"]
        info = run["info"]
        params = {entry["key"]: entry["value"] for entry in run["data"].get("params", [])}
        tags = {entry["key"]: entry["value"] for entry in run["data"].get("tags", [])}
        if (info["run_id"] != run_id or info["experiment_id"] != experiment
                or info["artifact_uri"] != links["mlflow_artifact_uri"]
                or params.get("dataset_sha256", fields["PreparedSha256"]) != fields["PreparedSha256"]
                or tags.get("cb-dataset", dataset) != dataset or tags.get("cb-scope", scope) != scope):
            raise ValueError("MLflow run does not match the evaluation manifest")
        if info["lifecycle_stage"] != "active":
            return {"status": "inactive_mlflow_run", "run": run_id}
        if info["status"] != "FINISHED" or not run["data"].get("metrics"):
            raise RuntimeError("MLflow evaluation is not finished with metrics yet")

        asset = self.ensure_asset(fields, dataset)
        asset_forms = metadata_forms(asset)
        run_url = f"{self.mlflow_ui}/#/experiments/{experiment}/runs/{run_id}?workspace=default"
        common = asset_forms["AssetCommonDetailsForm"]
        expected = {"cb-catalog-asset-id": asset["id"],
                    "cb-catalog-snapshot-sha256": fields["SnapshotManifestSha256"]}
        if any(key in tags and tags[key] != value for key, value in expected.items()):
            raise ValueError("MLflow run is already linked to a different catalog snapshot")
        if run_url not in common["readMe"] and tags.get("cb-catalog-asset-id") != asset["id"]:
            link = f"- [{campaign} · {unit}]({run_url})"
            common["readMe"] = readme_with_run(common["readMe"], link)
            asset_forms["SnapshotMetadata"]["MlflowEvaluationCount"] += 1
            revision, _ = self.form_types
            asset = self.datazone.create_asset_revision(
                domainIdentifier=self.domain, identifier=asset["id"], typeRevision=revision,
                clientToken=token("evaluation", asset["id"], asset["revision"], run_id),
                name=asset["name"], description=asset["description"], formsInput=self.forms(asset_forms),
                **({"glossaryTerms": asset["glossaryTerms"]} if asset.get("glossaryTerms") else {}),
            )
        listing = self.publish(asset)
        expected["cb-catalog-asset-url"] = (
            f"{self.portal}/projects/{self.project}/search/{listing}/latest/overview"
        )
        changes = [{"key": key, "value": value} for key, value in expected.items() if tags.get(key) != value]
        if changes:
            self.mlflow("log-batch", run_id=run_id, tags=changes)
        return {"status": "linked", "listing": listing, "asset": asset["id"], "revision": asset["revision"],
                "run": run_id, "tagged": bool(changes)}

    def process(self, event):
        detail = event.get("detail", {})
        if event.get("source") != "aws.s3" or event.get("detail-type") != "Object Created":
            return {"status": "ignored"}
        if detail.get("bucket", {}).get("name") != self.bucket:
            return {"status": "ignored"}
        obj = detail["object"]
        key, version = obj["key"], obj.get("version-id")
        if marker := MARKER.fullmatch(key):
            fields = self.snapshot(marker.groups(), version)
            asset = self.ensure_asset(fields, marker[1])
            return {"status": "published", "listing": self.publish(asset), "asset": asset["id"],
                    "revision": asset["revision"]}
        if final := FINAL.fullmatch(key):
            return self.evaluation(key, final, version)
        return {"status": "ignored"}


def handler(event, context):
    publisher = Publisher(
        boto3.Session(), bucket=os.environ["DATA_BUCKET"], domain=os.environ["DOMAIN_ID"],
        project=os.environ["PROJECT_ID"], portal=os.environ["PORTAL_URL"],
        mlflow_arn=os.environ["MLFLOW_ARN"], mlflow_ui=os.environ["MLFLOW_UI"],
    )
    for record in event["Records"]:
        result = publisher.process(json.loads(record["body"]))
        print(json.dumps({"message_id": record["messageId"], **result}))
