"""Validate Amazon States Language definitions offline, and optionally with the AWS API.

Offline checks (stdlib only): state types and required fields, transitions inside each scope,
reachability, terminal states, Choice/Retry/Catch shapes, JSONPath `.$` fields, substitution
placeholders, Lambda retries on transient errors, and project rules (no Fail state inside Map
item processors so one failed run never aborts its siblings; benchmark submissions are tagged
and reconciled before submit). `--service` calls ValidateStateMachineDefinition after replacing
`${...}` placeholders with syntactically valid ARNs (needs AWS credentials).

Usage: python scripts/aws/validate_asl.py [--service] infra/statemachines/*.asl.json
"""

import json
import re
import sys
from pathlib import Path

TYPES = {"Task", "Pass", "Choice", "Wait", "Succeed", "Fail", "Parallel", "Map"}
TERMINAL = {"Succeed", "Fail"}
LAMBDA_TRANSIENT = {
    "Lambda.ServiceException",
    "Lambda.AWSLambdaException",
    "Lambda.SdkClientException",
    "Lambda.TooManyRequestsException",
}
PLACEHOLDER = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
SAMPLE_SUBSTITUTIONS = {
    "ControlFunction": "arn:aws:lambda:us-east-1:123456789012:function:continual-bench-control:1",
    "JobQueue": "arn:aws:batch:us-east-1:123456789012:job-queue/continual-bench-jobs",
    "AcquisitionProject": "continual-bench-acquisition",
    "Project": "continual-bench",
}


def _json_paths(value, path, errors):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith(".$"):
                if not isinstance(item, str) or not (item.startswith("$") or item.startswith("States.")):
                    errors.append(f"{path}.{key}: '.$' fields need a JSONPath or intrinsic, got {item!r}")
            _json_paths(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json_paths(item, f"{path}[{index}]", errors)


def _retry_catch(name, state, scope_states, errors):
    for retry in state.get("Retry", []):
        if not retry.get("ErrorEquals"):
            errors.append(f"{name}: Retry without ErrorEquals")
    for catch in state.get("Catch", []):
        if not catch.get("ErrorEquals") or catch.get("Next") not in scope_states:
            errors.append(f"{name}: Catch must name errors and a Next in the same scope")


def validate_scope(definition: dict, where: str, errors: list, *, in_map=False):
    states = definition.get("States") or {}
    start = definition.get("StartAt")
    if start not in states:
        errors.append(f"{where}: StartAt {start!r} is not a state")
        return
    edges = {}
    for name, state in states.items():
        kind = state.get("Type")
        label = f"{where}/{name}"
        if kind not in TYPES:
            errors.append(f"{label}: unknown Type {kind!r}")
            continue
        targets = []
        if kind not in TERMINAL and kind != "Choice":
            if state.get("End") is True:
                if "Next" in state:
                    errors.append(f"{label}: both End and Next")
            elif state.get("Next") in states:
                targets.append(state["Next"])
            else:
                errors.append(f"{label}: Next {state.get('Next')!r} missing in scope")
        if kind == "Choice":
            choices = state.get("Choices") or []
            if not choices:
                errors.append(f"{label}: Choice without Choices")
            for choice in choices:
                if choice.get("Next") not in states:
                    errors.append(f"{label}: choice Next {choice.get('Next')!r} missing")
                targets.append(choice.get("Next"))
            if "Default" in state:
                if state["Default"] not in states:
                    errors.append(f"{label}: Default missing")
                targets.append(state["Default"])
        if kind == "Task":
            if not isinstance(state.get("Resource"), str) or not state["Resource"].startswith("arn:aws:"):
                errors.append(f"{label}: Task Resource must be an ARN")
            if state.get("Resource") == "arn:aws:states:::lambda:invoke":
                retried = {e for r in state.get("Retry", []) for e in r.get("ErrorEquals", [])}
                if not LAMBDA_TRANSIENT <= retried:
                    errors.append(f"{label}: Lambda task lacks retries for transient Lambda errors")
            if state.get("Resource", "").startswith("arn:aws:states:::batch:submitJob"):
                parameters = state.get("Parameters", {})
                if "Tags" not in parameters or not parameters.get("PropagateTags"):
                    errors.append(f"{label}: Batch submissions must carry and propagate cost/ownership tags")
        if kind == "Pass" and "ResultSelector" in state:
            errors.append(f"{label}: Pass does not support ResultSelector")
        if kind == "Fail" and in_map:
            errors.append(f"{label}: Fail inside a Map item processor aborts sibling runs")
        if kind == "Map":
            processor = state.get("ItemProcessor") or state.get("Iterator")
            if not processor:
                errors.append(f"{label}: Map without ItemProcessor")
            else:
                mode = (processor.get("ProcessorConfig") or {}).get("Mode", "INLINE")
                validate_scope(processor, f"{label}[{mode}]", errors, in_map=True)
        _retry_catch(label, state, states, errors)
        targets.extend(c["Next"] for c in state.get("Catch", []) if c.get("Next") in states)
        edges[name] = targets
    seen, frontier = set(), [start]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        frontier.extend(edges.get(name, []))
    for name in set(states) - seen:
        errors.append(f"{where}/{name}: unreachable")
    if not any(states[n].get("Type") in TERMINAL or states[n].get("End") for n in seen):
        errors.append(f"{where}: no reachable terminal state")


def project_rules(name: str, definition: dict, errors: list):
    text = json.dumps(definition)
    if "benchmark" in name:
        if "reconcile_unit" not in text or "reconcile_campaign" not in text:
            errors.append(
                f"{name}: benchmark workflow must reconcile before submitting and after the campaign"
            )
        if "submitJob.sync" not in text:
            errors.append(f"{name}: runs must be submitted with batch:submitJob.sync")
    if "ecs:runTask" in text or "ecs:createService" in text:
        errors.append(f"{name}: workers are Batch jobs, never ECS services or ad-hoc tasks")


def validate_file(path: Path, substitutions=SAMPLE_SUBSTITUTIONS) -> list[str]:
    text = path.read_text()
    errors = []
    unknown = set(PLACEHOLDER.findall(text)) - set(substitutions)
    if unknown:
        errors.append(f"{path.name}: undefined substitutions {sorted(unknown)}")
    definition = json.loads(text)
    _json_paths(definition, path.name, errors)
    validate_scope(definition, path.name, errors)
    project_rules(path.name, definition, errors)
    return errors


def substituted(path: Path) -> str:
    return PLACEHOLDER.sub(lambda m: SAMPLE_SUBSTITUTIONS[m.group(1)], path.read_text())


def service_validate(path: Path) -> list[str]:
    import boto3

    client = boto3.client("stepfunctions", region_name="us-east-1")
    response = client.validate_state_machine_definition(definition=substituted(path), type="STANDARD")
    return [
        f"{path.name}: {d['code']} {d.get('location', '')} {d['message']}"
        for d in response.get("diagnostics", [])
        if d["severity"] == "ERROR"
    ]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    service = "--service" in argv
    paths = [Path(p) for p in argv if p != "--service"]
    errors = []
    for path in paths:
        errors.extend(service_validate(path) if service else validate_file(path))
    for error in errors:
        print(error, file=sys.stderr)
    print(
        json.dumps(
            {
                "validated": [p.name for p in paths],
                "mode": "service" if service else "offline",
                "errors": len(errors),
            }
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
