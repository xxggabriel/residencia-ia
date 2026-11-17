"""One artifact layout and one concurrent uploader for every MLflow publisher."""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SKIP_SUFFIXES = (".lock",)
SKIP_PREFIXES = (".pending-", ".predictions-", ".model-", ".prepare-", ".zia-resume")
DEFAULT_UPLOAD_WORKERS = 8


def artifact_plan(output_dir, *, exclude: Iterable[str] = ()) -> list[tuple[Path, str]]:
    """Map run files to artifact paths exactly once.

    ``artifacts/<group>/…`` is published as ``<group>/…`` (``plots/``,
    ``data/``); every other file keeps its relative path. Temporary files and
    the ``exclude`` prefixes (evidence already published elsewhere, e.g. fit
    observations on child runs) are skipped. Earlier layouts uploaded the
    ``artifacts/`` tree twice and fit observations to both parent and child.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return []
    excluded = tuple(prefix.rstrip("/") + "/" for prefix in exclude)
    plan, seen = [], set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(SKIP_SUFFIXES):
            continue
        relative = path.relative_to(root)
        if any(part.startswith(SKIP_PREFIXES) for part in relative.parts):
            continue
        if (relative.as_posix() + "/").startswith(excluded):
            continue
        parts = (
            relative.parts[1:]
            if relative.parts[0] == "artifacts" and len(relative.parts) > 1
            else relative.parts
        )
        target = "/".join(parts)
        if target not in seen:
            seen.add(target)
            plan.append((path, target))
    return plan


def upload_concurrently(
    put: Callable[[Path, str], object], plan: list[tuple[Path, str]], *, workers: int = DEFAULT_UPLOAD_WORKERS
) -> list:
    """Upload independent files in parallel; the first failure is re-raised."""
    if not plan:
        return []
    workers = max(1, min(workers, len(plan)))
    if workers == 1:
        return [put(path, target) for path, target in plan]
    with ThreadPoolExecutor(workers, thread_name_prefix="mlflow-artifacts") as pool:
        return list(pool.map(lambda item: put(*item), plan))
