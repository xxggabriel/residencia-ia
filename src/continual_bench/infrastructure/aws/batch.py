"""AWS Batch and ECS lookups used by reconciliation; jobs are finite, never ECS services."""

from .session import client as make_client

ACTIVE = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
TERMINAL = ("SUCCEEDED", "FAILED")


def task_cluster(task_arn: str) -> str:
    """`arn:aws:ecs:<region>:<account>:task/<cluster>/<id>` → cluster name."""
    resource = task_arn.split(":", 5)[5]
    parts = resource.split("/")
    if len(parts) != 3 or parts[0] != "task":
        raise ValueError(f"unexpected ECS task ARN format: {task_arn}")
    return parts[1]


class BatchJobs:
    def __init__(self, queue: str, *, client=None, ecs=None):
        self.queue = queue
        self.client = client or make_client("batch")
        self._ecs = ecs

    @property
    def ecs(self):
        if self._ecs is None:
            self._ecs = make_client("ecs")
        return self._ecs

    def find(self, name_pattern: str) -> list[dict]:
        """All jobs (every status) whose name matches; a trailing `*` is a prefix wildcard."""
        jobs, token = [], None
        while True:
            arguments = {"jobQueue": self.queue, "filters": [{"name": "JOB_NAME", "values": [name_pattern]}]}
            if token:
                arguments["nextToken"] = token
            page = self.client.list_jobs(**arguments)
            jobs.extend(page.get("jobSummaryList", []))
            token = page.get("nextToken")
            if not token:
                return sorted(jobs, key=lambda job: job.get("createdAt", 0))

    def describe(self, job_ids) -> list[dict]:
        job_ids, found = list(job_ids), []
        for offset in range(0, len(job_ids), 100):
            found.extend(self.client.describe_jobs(jobs=job_ids[offset : offset + 100])["jobs"])
        return found

    def submit(self, *, name: str, definition: str, parameters: dict, tags: dict) -> str:
        return self.client.submit_job(
            jobName=name,
            jobQueue=self.queue,
            jobDefinition=definition,
            parameters=parameters,
            tags=tags,
            propagateTags=True,
        )["jobId"]

    def terminate(self, job_id: str, reason: str) -> None:
        # TerminateJob cancels queued jobs and stops STARTING/RUNNING ones.
        self.client.terminate_job(jobId=job_id, reason=reason[:256])

    def task_states(self, jobs) -> dict[str, str]:
        """ECS lastStatus for every attempt task of the given described jobs."""
        by_cluster: dict[str, list[str]] = {}
        for job in jobs:
            for attempt in job.get("attempts", []):
                arn = attempt.get("container", {}).get("taskArn")
                if arn:
                    by_cluster.setdefault(task_cluster(arn), []).append(arn)
            arn = job.get("container", {}).get("taskArn")
            if arn and arn not in by_cluster.get(task_cluster(arn), []):
                by_cluster.setdefault(task_cluster(arn), []).append(arn)
        states = {}
        for cluster, arns in by_cluster.items():
            for offset in range(0, len(arns), 100):
                response = self.ecs.describe_tasks(cluster=cluster, tasks=arns[offset : offset + 100])
                states.update({task["taskArn"]: task["lastStatus"] for task in response.get("tasks", [])})
                # Tasks stopped long ago age out of ECS; report them as such instead of guessing.
                states.update(
                    {
                        failure["arn"]: "MISSING_" + failure.get("reason", "UNKNOWN")
                        for failure in response.get("failures", [])
                    }
                )
        return states
