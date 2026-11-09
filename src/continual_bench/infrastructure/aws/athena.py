"""Bounded Athena query runner for results and cost queries (workgroup enforces output location)."""

import time

from .session import client as make_client


def run_query(
    sql: str, *, workgroup: str, database: str, client=None, timeout_seconds=300, poll_seconds=1.0
) -> list[dict]:
    client = client or make_client("athena")
    execution = client.start_query_execution(
        QueryString=sql,
        WorkGroup=workgroup,
        QueryExecutionContext={"Database": database},
    )["QueryExecutionId"]
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = client.get_query_execution(QueryExecutionId=execution)["QueryExecution"]["Status"]
        if status["State"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        if time.monotonic() > deadline:
            client.stop_query_execution(QueryExecutionId=execution)
            raise TimeoutError(f"Athena query exceeded {timeout_seconds}s: {execution}")
        time.sleep(poll_seconds)
    if status["State"] != "SUCCEEDED":
        raise RuntimeError(f"Athena query {execution} {status['State']}: {status.get('StateChangeReason')}")
    rows, header, token = [], None, None
    while True:
        page = client.get_query_results(QueryExecutionId=execution, **({"NextToken": token} if token else {}))
        for row in page["ResultSet"]["Rows"]:
            values = [cell.get("VarCharValue") for cell in row["Data"]]
            if header is None:
                header = values
            else:
                rows.append(dict(zip(header, values, strict=True)))
        token = page.get("NextToken")
        if not token:
            return rows
