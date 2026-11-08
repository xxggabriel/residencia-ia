"""Install as a trusted module, then set plugins: [examples.custom_plugin:register]."""

from continual_bench.core.contracts import Decision


class EveryTenPolicy:
    def __init__(self, **_validated_policy_fields):
        self.last = 0

    def decide(self, context):
        if context.labeled - self.last >= 10:
            return Decision("learn_one", "custom_every_ten")
        return Decision("none", "waiting")

    def updated(self, labeled):
        self.last = labeled


class RunningMeanRegressor:
    """Regression baseline: predicts the mean of every target learned so far."""

    def __init__(self):
        self.count, self.total = 0, 0.0

    def predict_one(self, x):
        return self.total / self.count if self.count else 0.0

    def learn_one(self, x, y):
        self.count += 1
        self.total += y


def register(registry):
    registry.register("policy", "every_ten", EveryTenPolicy)
    # Models declare the target types they serve; evaluators reject other datasets.
    registry.register("model", "running_mean", RunningMeanRegressor, tasks=("regression",))
