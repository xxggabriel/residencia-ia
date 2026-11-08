"""River metric suites per target type for the typed and epoch evaluators."""

from river import metrics

from continual_bench.core.tasks import REGRESSION


def metric_set(task: str) -> dict:
    return ({"mae": metrics.MAE(), "rmse": metrics.RMSE()} if task == REGRESSION else
            {"accuracy": metrics.Accuracy(), "macro_f1": metrics.MacroF1()})
