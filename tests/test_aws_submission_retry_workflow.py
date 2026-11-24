"""Execute the actual ASL item processors with infrastructure protocol fixtures.

Only the Lambda/Batch service boundaries are substituted. Reconciliation and attempt
recording use the real lifecycle and local object store. No models or metric values
are generated. The small interpreter implements the JSONPath constructs used here;
AWS service-side validation remains part of CI.
"""

import copy
import json
import re
from collections import deque
from pathlib import Path

import pytest

from continual_bench.application.cloud import layout, lifecycle
from continual_bench.application.cloud.planning import INLINE_LIMIT
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json, put_json

ROOT = Path(__file__).parents[1]
DEFINITION = ROOT / "infra/statemachines/benchmark.asl.json"
PROCESSORS = ("InlineRuns", "DistributedRuns")


class ServiceFailure(Exception):
    def __init__(self, error, cause):
        self.payload = {"Error": error, "Cause": cause}


class JobFacts:
    """Batch metadata only, including terminal jobs left by earlier evaluations."""

    def __init__(self):
        self.jobs = {}

    def find(self, name):
        return [copy.deepcopy(job) for job in self.jobs.values() if job["jobName"] == name]

    def describe(self, ids):
        return [copy.deepcopy(self.jobs[ident]) for ident in ids]

    def terminate(self, ident, reason):
        self.jobs[ident]["status"] = "FAILED"
        self.jobs[ident]["statusReason"] = reason


class Services:
    def __init__(self, directory, outcomes=(), *, max_attempts=2, run_id="unit"):
        self.store = LocalObjectStore(directory)
        self.jobs = JobFacts()
        self.outcomes = deque(outcomes)
        self.item = {
            "campaign_id": "submission-contract",
            "run_id": run_id,
            "job_name": f"cb-submission-contract-{run_id}",
            "unit_uri": f"s3://fixture/{run_id}/unit.json",
            "job_definition": "arn:aws:batch:us-east-1:123456789012:job-definition/fixture:1",
            "max_attempts": max_attempts,
            "run_timeout_seconds": 3600,
            "task_timeout_seconds": 3900,
        }
        self.clock = 1_700_000_000
        self.submit_calls = 0
        self.actions = []
        self.waits = []
        self.on_wait = None

    def add_job(self, status):
        ident = f"job-{len(self.jobs.jobs) + 1}"
        self.jobs.jobs[ident] = {
            "jobId": ident,
            "jobName": self.item["job_name"],
            "jobDefinition": self.item["job_definition"],
            "status": status,
            "createdAt": self.clock * 1000,
            "startedAt": self.clock * 1000 + 1,
            "stoppedAt": self.clock * 1000 + 2 if status in ("FAILED", "SUCCEEDED") else None,
            "container": {
                "resourceRequirements": [
                    {"type": "VCPU", "value": "2"},
                    {"type": "MEMORY", "value": "4096"},
                ]
            },
        }
        return ident

    def publish_fixture_final(self):
        # Existence is the lifecycle publication contract; this contains no model or metrics.
        put_json(
            self.store,
            layout.final_key(self.item["campaign_id"], self.item["run_id"]),
            {"infrastructure_publication_fixture": True},
        )

    def invoke(self, parameters):
        event = parameters["Payload"]
        action = event["action"]
        self.actions.append(action)
        if action == "reconcile_unit":
            payload = lifecycle.reconcile_unit(self.store, self.jobs, event["item"])
        elif action == "record_attempt":
            payload = lifecycle.record_attempt(
                self.store,
                self.jobs,
                event["item"],
                error=event.get("error"),
                clock=lambda: self.clock,
            )
        elif action == "describe_job":
            payload = lifecycle.describe_job(self.jobs, event["job_id"])
        else:
            raise AssertionError(f"Unexpected control action: {action}")
        return {"Payload": payload}

    def submit(self):
        self.submit_calls += 1
        self.clock += 1
        outcome = self.outcomes.popleft() if self.outcomes else "reject"
        if isinstance(outcome, tuple):
            kind, error = outcome
        else:
            kind, error = outcome, "Batch.ClientException"
        if kind == "reject":
            raise ServiceFailure(error, "JobQueue not ENABLED; no job created")
        status = "RUNNING" if kind == "accepted_error" else "FAILED" if kind == "failed" else "SUCCEEDED"
        ident = self.add_job(status)
        if kind == "accepted_error":
            raise ServiceFailure("States.Timeout", "Submission response unavailable")
        if kind == "failed":
            # The error name alone must not turn this real evaluation into a submission failure.
            raise ServiceFailure(error, json.dumps({"JobId": ident, "Status": "FAILED"}))
        if kind == "unpublished":
            return {"JobId": ident, "Status": "SUCCEEDED"}
        assert kind == "success"
        self.publish_fixture_final()
        return {"JobId": ident, "Status": "SUCCEEDED"}

    def wait(self, seconds, state_name):
        self.waits.append(seconds)
        self.clock += seconds
        if self.on_wait:
            self.on_wait(self, state_name)

    def recorded_errors(self):
        prefix = f"{layout.run_prefix(self.item['campaign_id'], self.item['run_id'])}/events/"
        return [get_json(self.store, key)["workflow_error"] for key in self.store.list(prefix)]


def path_value(expression, data, context=None):
    if expression == "$":
        return copy.deepcopy(data)
    source = context if expression.startswith("$$.") else data
    assert expression.startswith(("$.", "$$.")), expression
    for part in expression.split(".")[1:]:
        source = source[part]
    return copy.deepcopy(source)


def render(template, data, context=None):
    if isinstance(template, list):
        return [render(value, data, context) for value in template]
    if not isinstance(template, dict):
        return copy.deepcopy(template)
    result = {}
    for key, value in template.items():
        if key.endswith(".$"):
            match = re.fullmatch(r"States\.MathAdd\(([^,]+),\s*([^)]+)\)", value)
            if match:
                operands = [
                    path_value(part.strip(), data, context)
                    if part.strip().startswith("$")
                    else int(part.strip())
                    for part in match.groups()
                ]
                result[key[:-2]] = sum(operands)
            else:
                result[key[:-2]] = path_value(value, data, context)
        else:
            result[key] = render(value, data, context)
    return result


def store_result(data, path, value):
    if path is None:
        return data
    if path == "$":
        return copy.deepcopy(value)
    output = copy.deepcopy(data)
    target = output
    parts = path.split(".")[1:]
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = copy.deepcopy(value)
    return output


def matches(choice, data):
    value = path_value(choice["Variable"], data)
    for operator in ("StringEquals", "BooleanEquals", "NumericGreaterThanEquals"):
        if operator in choice:
            expected = choice[operator]
            return value >= expected if operator == "NumericGreaterThanEquals" else value == expected
    if "NumericGreaterThanPath" in choice:
        return value > path_value(choice["NumericGreaterThanPath"], data)
    raise AssertionError(f"Unexpected Choice rule: {choice}")


def execute_item(processor_name, services, *, definition=None, transition_limit=250):
    definition = definition or json.loads(DEFINITION.read_text())
    fan_out = definition["States"][processor_name]
    processor = fan_out["ItemProcessor"]
    parent = {"planned": {"plan": {"control_function": "fixture-control"}}}
    data = render(fan_out["ItemSelector"], parent, {"Map": {"Item": {"Value": services.item}}})
    current, trace = processor["StartAt"], []
    for _ in range(transition_limit):
        state = processor["States"][current]
        kind = state["Type"]
        trace.append((current, kind))
        if kind == "Choice":
            current = next(
                (rule["Next"] for rule in state["Choices"] if matches(rule, data)), state.get("Default")
            )
            assert current
            continue
        if kind == "Wait":
            seconds = state.get("Seconds")
            seconds = path_value(state["SecondsPath"], data) if seconds is None else seconds
            services.wait(seconds, current)
        elif kind == "Pass":
            value = render(state["Parameters"], data) if "Parameters" in state else state["Result"]
            data = store_result(data, state.get("ResultPath", "$"), value)
        else:
            assert kind == "Task"
            try:
                parameters = render(state["Parameters"], data)
                value = (
                    services.invoke(parameters)
                    if state["Resource"].endswith("lambda:invoke")
                    else services.submit()
                )
            except ServiceFailure as error:
                catch = next(
                    rule
                    for rule in state["Catch"]
                    if "States.ALL" in rule["ErrorEquals"] or error.payload["Error"] in rule["ErrorEquals"]
                )
                data = store_result(data, catch.get("ResultPath", "$"), error.payload)
                current = catch["Next"]
                continue
            if "ResultSelector" in state:
                value = render(state["ResultSelector"], value)
            data = store_result(data, state.get("ResultPath", "$"), value)
        if state.get("End"):
            return data, trace
        current = state["Next"]
    raise AssertionError("ASL item exceeded the transition budget without reaching a terminal state")


@pytest.mark.parametrize("processor", PROCESSORS)
@pytest.mark.parametrize("error", ["Batch.ClientException", "Batch.ServerException", "States.TaskFailed"])
def test_no_job_submission_failures_terminate_and_keep_error_evidence(tmp_path, processor, error):
    services = Services(tmp_path, [("reject", error)] * 5, max_attempts=9)
    outcome, trace = execute_item(processor, services)
    assert outcome == {"run_id": "unit", "status": "FAILED"}
    assert services.submit_calls == 5 and not services.jobs.jobs
    assert services.waits == [30, 60, 120, 240]
    assert services.item["max_attempts"] == 9
    errors = services.recorded_errors()
    assert len(errors) == 5 and all(row["Error"] == error for row in errors)
    assert len(trace) < 100


@pytest.mark.parametrize("processor", PROCESSORS)
def test_rejection_after_historical_job_is_not_mistaken_for_a_new_job(tmp_path, processor):
    services = Services(tmp_path, max_attempts=3)
    previous = services.add_job("FAILED")
    outcome, _ = execute_item(processor, services)
    assert outcome["status"] == "FAILED" and services.submit_calls == 5
    assert services.waits == [30, 60, 120, 240]
    assert set(services.jobs.jobs) == {previous}
    assert services.recorded_errors() and services.item["max_attempts"] == 3


@pytest.mark.parametrize("processor", PROCESSORS)
def test_real_failed_job_with_client_exception_keeps_the_job_attempt_budget(tmp_path, processor):
    services = Services(tmp_path, [("failed", "Batch.ClientException")] * 2, max_attempts=2)
    outcome, _ = execute_item(processor, services)
    assert outcome["status"] == "FAILED"
    assert services.submit_calls == len(services.jobs.jobs) == 2
    assert services.waits == []
    assert len(services.recorded_errors()) == 2


@pytest.mark.parametrize("processor", PROCESSORS)
def test_new_real_job_keeps_the_global_no_job_budget(tmp_path, processor):
    sequence = ["reject"] * 4 + [("failed", "Batch.ClientException")] + ["reject"] * 4 + ["success"]
    services = Services(tmp_path, sequence, max_attempts=2)
    outcome, _ = execute_item(processor, services)
    assert outcome["status"] == "FAILED"
    assert services.submit_calls == 6 and len(services.jobs.jobs) == 1
    assert services.waits == [30, 60, 120, 240]
    assert services.item["max_attempts"] == 2


@pytest.mark.parametrize("processor", PROCESSORS)
def test_real_job_retry_uses_remaining_global_submission_budget(tmp_path, processor):
    sequence = ["reject"] * 2 + [("failed", "Batch.ClientException")] + ["reject"] * 2 + ["success"]
    services = Services(tmp_path, sequence, max_attempts=2)
    outcome, _ = execute_item(processor, services)
    assert outcome["status"] == "SUCCEEDED"
    assert services.submit_calls == 6 and len(services.jobs.jobs) == 2
    assert services.waits == [30, 60, 120, 240]
    assert services.item["max_attempts"] == 2


@pytest.mark.parametrize("processor", PROCESSORS)
def test_exit_zero_without_final_keeps_publication_failure_retry_semantics(tmp_path, processor):
    sequence = ["reject"] * 2 + ["unpublished"] + ["reject"] * 2 + ["success"]
    services = Services(tmp_path, sequence, max_attempts=2)
    outcome, trace = execute_item(processor, services)
    assert outcome["status"] == "SUCCEEDED"
    assert services.submit_calls == 6 and len(services.jobs.jobs) == 2
    assert services.waits == [30, 60, 120, 240]
    assert any(name.endswith("ResetAfterRecordedJob") for name, _ in trace)


@pytest.mark.parametrize("processor", PROCESSORS)
def test_transient_rejection_recovers_without_consuming_evaluation_attempt(tmp_path, processor):
    services = Services(tmp_path, ["reject", "reject", "success"], max_attempts=1)
    outcome, _ = execute_item(processor, services)
    assert outcome["status"] == "SUCCEEDED"
    assert services.submit_calls == 3 and len(services.jobs.jobs) == 1
    assert services.waits == [30, 60]


@pytest.mark.parametrize("processor", PROCESSORS)
def test_reconcile_after_wait_finds_a_late_job_without_duplicate_submission(tmp_path, processor):
    services = Services(tmp_path, ["reject"], max_attempts=1)

    def late_job(current, state):
        if state.endswith("WaitToResubmit"):
            current.add_job("RUNNING")
        elif state.endswith("WaitForExistingJob"):
            for job in current.jobs.jobs.values():
                job["status"] = "SUCCEEDED"
            current.publish_fixture_final()

    services.on_wait = late_job
    outcome, _ = execute_item(processor, services)
    assert outcome["status"] == "SUCCEEDED"
    assert services.submit_calls == len(services.jobs.jobs) == 1
    assert services.waits == [30, 60]


@pytest.mark.parametrize("processor", PROCESSORS)
def test_accepted_job_after_lost_response_keeps_existing_job_wait_and_retry(tmp_path, processor):
    services = Services(tmp_path, ["accepted_error", "success"], max_attempts=2)
    services.jobs.terminate = lambda _ident, _reason: None  # Batch termination is asynchronous.

    def finish_failed(current, _state):
        for job in current.jobs.jobs.values():
            job["status"] = "FAILED"

    services.on_wait = finish_failed
    outcome, _ = execute_item(processor, services)
    assert outcome["status"] == "SUCCEEDED"
    assert services.submit_calls == len(services.jobs.jobs) == 2
    assert services.waits == [60]


@pytest.mark.parametrize("processor", PROCESSORS)
def test_success_reuse_and_existing_job_do_not_submit_extra_jobs(tmp_path, processor):
    fresh = Services(tmp_path / "fresh", ["success"])
    assert execute_item(processor, fresh)[0]["status"] == "SUCCEEDED"
    assert fresh.submit_calls == 1 and not fresh.waits
    reused = Services(tmp_path / "reused")
    reused.publish_fixture_final()
    assert execute_item(processor, reused)[0]["status"] == "REUSED"
    assert reused.submit_calls == 0 and not reused.waits
    existing = Services(tmp_path / "existing")
    existing.add_job("RUNNING")

    def finish(current, _state):
        for job in current.jobs.jobs.values():
            job["status"] = "SUCCEEDED"
        current.publish_fixture_final()

    existing.on_wait = finish
    assert execute_item(processor, existing)[0]["status"] == "SUCCEEDED"
    assert existing.submit_calls == 0 and existing.waits == [60]


def event_bound(trace):
    """Conservative history allowance for immediate service outcomes in these regressions.

    Control Lambda calls succeed: this does not bound histories with additional
    Lambda-service errors or repeated polling of a long-running existing job.
    """
    return sum(12 if kind == "Task" else 4 for _, kind in trace)


def test_all_inline_no_job_items_have_a_finite_history_bound_and_independent_budgets(tmp_path):
    traces = []
    for index in range(INLINE_LIMIT):
        service = Services(tmp_path / str(index), run_id=f"unit-{index}")
        outcome, trace = execute_item("InlineRuns", service)
        assert outcome["status"] == "FAILED" and service.submit_calls == 5
        traces.append(trace)
    # Conservative allowance for this reproduced fault: 12 history events per Task
    # execution, four per other state, plus 1,000 for the outer Map/control states.
    # Control Lambda calls succeed here; additional unrelated service faults are not simulated.
    history_bound = 1_000 + sum(event_bound(trace) for trace in traces)
    assert INLINE_LIMIT == 40 and history_bound < 25_000
    sibling = Services(tmp_path / "healthy-sibling", ["success"], run_id="healthy-sibling")
    assert execute_item("InlineRuns", sibling)[0]["status"] == "SUCCEEDED"


@pytest.mark.parametrize("processor", PROCESSORS)
@pytest.mark.parametrize("items", [21, INLINE_LIMIT])
def test_full_request_mixed_no_job_and_real_failure_stays_below_history_limit(tmp_path, processor, items):
    sequence = ["reject"] * 4 + [("failed", "Batch.ClientException")]
    sequence += ["reject"] * 4 + [("failed", "Batch.ClientException")]
    services = Services(tmp_path, sequence, max_attempts=2)
    outcome, trace = execute_item(processor, services)
    assert outcome["status"] == "FAILED"
    # The fifth global NoJob rejection terminates before the planned second real job.
    assert services.submit_calls == 6 and len(services.jobs.jobs) == 1
    assert services.waits == [30, 60, 120, 240]
    assert 1_000 + items * event_bound(trace) < 25_000


def terminal_rejection_and_failed_job_orders(no_jobs=0, real_jobs=0, sequence=()):
    """All finite orderings until either five NoJob errors or two real failed jobs."""
    if no_jobs == 5 or real_jobs == 2:
        yield sequence
        return
    yield from terminal_rejection_and_failed_job_orders(no_jobs + 1, real_jobs, (*sequence, "reject"))
    yield from terminal_rejection_and_failed_job_orders(
        no_jobs, real_jobs + 1, (*sequence, ("failed", "Batch.ClientException"))
    )


@pytest.mark.parametrize("processor", PROCESSORS)
def test_global_budget_bounds_all_mixed_terminal_orders_for_max_attempts_two(tmp_path, processor):
    bounds = []
    sequences = list(terminal_rejection_and_failed_job_orders())
    assert len(sequences) == 21
    for index, sequence in enumerate(sequences):
        services = Services(tmp_path / str(index), sequence, max_attempts=2)
        outcome, trace = execute_item(processor, services)
        assert outcome["status"] == "FAILED"
        real_jobs = len(services.jobs.jobs)
        no_jobs = services.submit_calls - real_jobs
        assert no_jobs <= 5 and real_jobs <= 2 and services.submit_calls <= 6
        assert no_jobs == 5 or real_jobs == 2
        assert services.item["max_attempts"] == 2
        bounds.append(event_bound(trace))
    assert 1_000 + 21 * max(bounds) < 25_000
    assert 1_000 + INLINE_LIMIT * max(bounds) < 25_000
