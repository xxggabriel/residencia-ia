"""S3 implementation of the object store port.

Small documents use `PutObject` with `If-None-Match: *`. Large files use a multipart upload
with per-part SHA-256 checksums and a conditional `CompleteMultipartUpload`, so a completion
never replaces an existing object. The full-object SHA-256 computed locally is stored as user
metadata (`sha256`) because S3 multipart checksums are composite.
"""

import hashlib
from pathlib import Path

from continual_bench.infrastructure.object_store import (
    ObjectExists,
    StoredObject,
    clean_key,
    sha256_path,
    split_uri,
)

from .session import client as make_client
from .session import error_code

PART_BYTES = 64 * 1024**2
SINGLE_PUT_LIMIT = 64 * 1024**2
CONFLICT_CODES = {"PreconditionFailed", "ConditionalRequestConflict"}


def _checksum(data: bytes) -> str:
    import base64

    return base64.b64encode(hashlib.sha256(data).digest()).decode()


class S3ObjectStore:
    def __init__(self, bucket: str, *, client=None, kms_key_id: str | None = None):
        self.bucket = bucket
        self.client = client or make_client("s3")
        self.kms_key_id = kms_key_id

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{clean_key(key)}"

    def _encryption(self) -> dict:
        if self.kms_key_id:
            return {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": self.kms_key_id,
                "BucketKeyEnabled": True,
            }
        return {}

    def put_bytes(
        self, key, data: bytes, *, if_absent=False, metadata=None, content_type=None
    ) -> StoredObject:
        digest = hashlib.sha256(data).hexdigest()
        arguments = {
            "Bucket": self.bucket,
            "Key": clean_key(key),
            "Body": data,
            "ChecksumSHA256": _checksum(data),
            "Metadata": {**(metadata or {}), "sha256": digest},
            **self._encryption(),
        }
        if content_type:
            arguments["ContentType"] = content_type
        if if_absent:
            arguments["IfNoneMatch"] = "*"
        try:
            response = self.client.put_object(**arguments)
        except Exception as error:
            if error_code(error) in CONFLICT_CODES:
                raise ObjectExists(key) from error
            raise
        return StoredObject(
            key, self.uri(key), len(data), response.get("VersionId"), digest, arguments["Metadata"]
        )

    def put_file(
        self, key, path, *, if_absent=False, metadata=None, sha256=None, content_type=None
    ) -> StoredObject:
        path = Path(path)
        size = path.stat().st_size
        if size <= SINGLE_PUT_LIMIT:
            data = path.read_bytes()
            if sha256 and hashlib.sha256(data).hexdigest() != sha256:
                raise ValueError(f"local file changed before upload: {path}")
            return self.put_bytes(
                key, data, if_absent=if_absent, metadata=metadata, content_type=content_type
            )
        sha256 = sha256 or sha256_path(path)
        digest = hashlib.sha256()
        meta = dict(metadata or {})
        if if_absent and self.head(key) is not None:
            raise ObjectExists(key)
        upload = self.client.create_multipart_upload(
            Bucket=self.bucket,
            Key=clean_key(key),
            ChecksumAlgorithm="SHA256",
            Metadata={**meta, "sha256": sha256},
            **({"ContentType": content_type} if content_type else {}),
            **self._encryption(),
        )
        upload_id, parts = upload["UploadId"], []
        try:
            with path.open("rb") as stream:
                number = 1
                while chunk := stream.read(PART_BYTES):
                    digest.update(chunk)
                    response = self.client.upload_part(
                        Bucket=self.bucket,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=number,
                        Body=chunk,
                        ChecksumAlgorithm="SHA256",
                        ChecksumSHA256=_checksum(chunk),
                    )
                    parts.append(
                        {
                            "PartNumber": number,
                            "ETag": response["ETag"],
                            "ChecksumSHA256": response.get("ChecksumSHA256", _checksum(chunk)),
                        }
                    )
                    number += 1
            computed = digest.hexdigest()
            if computed != sha256:
                raise ValueError(f"local file changed during upload: {path}")
            arguments = {
                "Bucket": self.bucket,
                "Key": key,
                "UploadId": upload_id,
                "MultipartUpload": {"Parts": parts},
            }
            if if_absent:
                arguments["IfNoneMatch"] = "*"
            try:
                response = self.client.complete_multipart_upload(**arguments)
            except Exception as error:
                if error_code(error) in CONFLICT_CODES:
                    raise ObjectExists(key) from error
                raise
        except BaseException:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
            raise
        return StoredObject(
            key, self.uri(key), size, response.get("VersionId"), computed, {**meta, "sha256": computed}
        )

    def head(self, key) -> StoredObject | None:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=clean_key(key))
        except Exception as error:
            if error_code(error) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        metadata = response.get("Metadata", {})
        return StoredObject(
            key,
            self.uri(key),
            response["ContentLength"],
            response.get("VersionId"),
            metadata.get("sha256"),
            metadata,
        )

    def get_bytes(self, key) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=clean_key(key))["Body"].read()
        except Exception as error:
            if error_code(error) in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(key) from error
            raise

    def download(self, key, destination) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, clean_key(key), str(destination))
        return destination

    def download_uri(self, uri, destination) -> Path:
        bucket, key = split_uri(uri)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, key, str(destination))
        return destination

    def list(self, prefix: str = ""):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield item["Key"]
