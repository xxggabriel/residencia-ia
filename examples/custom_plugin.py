"""Install as a trusted module, then set plugins: [examples.custom_plugin:register]."""

from continual_bench.domain.contracts import Decision


class EveryTenPolicy:
    def __init__(self, **_validated_policy_fields):
        self.last = 0

    def decide(self, context):
        if context.labeled - self.last >= 10:
            return Decision("learn_one", "custom_every_ten")
        return Decision("none", "waiting")

    def updated(self, labeled):
        self.last = labeled


def register(registry):
    registry.register("policy", "every_ten", EveryTenPolicy)
