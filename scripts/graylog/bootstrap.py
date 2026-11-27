"""Provision the project GELF input, stream and read token through Graylog's API."""

import argparse
import base64
import json
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path

INPUT = "Residência em IA GELF"
INPUT_TYPE = "org.graylog2.inputs.gelf.http.GELFHttpInput"
STREAM = "Residência em IA — eventos operacionais"
READER = "residencia-airflow-reader"


def api(url, credential, path, body=None):
    encoded = base64.b64encode(credential.encode()).decode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Requested-By": "residencia-bootstrap",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Graylog returned {error.code} for {path}: {error.read(1024).decode()}") from None
    return json.loads(data) if data else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:19000")
    parser.add_argument("--admin-password-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--environment", default="homelab")
    args = parser.parse_args()

    def admin(path, body=None):
        return api(args.url, "admin:" + args.admin_password_file.read_text().strip(), path, body)

    inputs = admin("system/inputs")["inputs"]
    existing_input = next((item for item in inputs if item["title"] == INPUT), None)
    if not existing_input:
        config = admin(f"system/inputs/types/{INPUT_TYPE}")["requested_configuration"]
        values = {key: field["default_value"] for key, field in config.items()}
        values.update(enable_bulk_receiving=True, number_worker_threads=2)
        admin("system/inputs", {"title": INPUT, "type": INPUT_TYPE, "global": True, "configuration": values})

    streams = admin("streams")["streams"]
    existing_stream = next((item for item in streams if item["title"] == STREAM), None)
    if not existing_stream:
        default = next(item for item in streams if item["is_default"])
        result = admin(
            "streams",
            {
                "entity": {
                    "title": STREAM,
                    "description": f"Eventos operacionais da Residência em IA ({args.environment})",
                    "index_set_id": default["index_set_id"],
                    "matching_type": "AND",
                    "remove_matches_from_default_stream": True,
                    "rules": [
                        {"field": "service", "type": 1, "value": "benchmark-worker", "inverted": False},
                        {"field": "environment", "type": 1, "value": args.environment, "inverted": False},
                    ],
                }
            },
        )
        stream_id = result["stream_id"]
        admin(f"streams/{stream_id}/resume", {})
    else:
        stream_id = existing_stream["id"]
        fields = {rule["field"]: rule["value"] for rule in existing_stream["rules"] if rule["type"] == 1}
        if fields.get("environment") != args.environment or fields.get("service") != "benchmark-worker":
            raise SystemExit("Existing project stream filters differ from the requested environment/service.")
        if existing_stream["disabled"]:
            admin(f"streams/{stream_id}/resume", {})

    def find_user():
        return next((item for item in admin("users")["users"] if item["username"] == READER), None)

    user = find_user()
    if not user:
        admin(
            "users",
            {
                "username": READER,
                "first_name": "Residência",
                "last_name": "Airflow",
                "email": "airflow@example.invalid",
                "password": secrets.token_urlsafe(36),
                "permissions": [],
                "roles": ["Reader"],
                "service_account": True,
            },
        )
        user = find_user()
    if set(user["roles"]) != {"Reader"}:
        raise SystemExit("The existing Airflow user must have only the Reader role.")
    user_id = user["id"]
    share_path = f"authz/shares/entities/grn::::stream:{stream_id}"
    grants = admin(f"{share_path}/prepare", {"selected_grantee_capabilities": {}})
    selected = grants["selected_grantee_capabilities"]
    grantee = f"grn::::user:{user_id}"
    if selected.get(grantee) != "view":
        admin(share_path, {"selected_grantee_capabilities": {**selected, grantee: "view"}})

    token = args.token_file.read_text().strip() if args.token_file.exists() else ""
    if not token:
        name = "airflow-logs-" + secrets.token_hex(4)
        token = admin(f"users/{user_id}/tokens/{name}", {})["token"]
    readable = api(args.url, token + ":token", "streams")["streams"]
    if {item["id"] for item in readable} != {stream_id}:
        raise SystemExit("The Airflow token must see exactly the project stream.")

    args.token_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not args.token_file.exists() or not args.token_file.read_text().strip():
        temporary = args.token_file.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "w") as output:
            output.write(token + "\n")
        # Airflow runs with UID 50000. Keep the enclosing host directory private.
        temporary.chmod(0o444)
        temporary.replace(args.token_file)
    print(f"BENCH_GRAYLOG_STREAM={stream_id}")


if __name__ == "__main__":
    main()
