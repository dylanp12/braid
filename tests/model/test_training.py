import math

import pytest
import torch

from braid.model.training import ObjectiveWeights, censored_exponential_nll


def test_objective_mix_is_exactly_65_15_10_10() -> None:
    weights = ObjectiveWeights()
    total = weights.combine(
        patch=torch.tensor(1.0),
        schema_episode=torch.tensor(2.0),
        reconstruction=torch.tensor(3.0),
        contrastive=torch.tensor(4.0),
    )
    assert torch.isclose(total, torch.tensor(1.65))


def test_censored_interval_uses_survival_term_without_event_density() -> None:
    log_rate = torch.tensor([math.log(0.5), math.log(0.5)])
    duration = torch.tensor([2.0, 2.0])
    observed = torch.tensor([True, False])
    losses = censored_exponential_nll(log_rate, duration, observed, reduction="none")
    assert torch.isclose(losses[0], torch.tensor(1.0 - math.log(0.5)))
    assert torch.isclose(losses[1], torch.tensor(1.0))


def test_rate_clamp_does_not_leave_an_unbounded_log_rate_gradient() -> None:
    log_rate = torch.tensor([100.0], requires_grad=True)
    loss = censored_exponential_nll(
        log_rate,
        torch.tensor([1.0]),
        torch.tensor([True]),
    )

    loss.backward()

    assert log_rate.grad is not None
    assert log_rate.grad.item() == 0.0


def test_explicit_time_scale_normalizes_raw_durations_and_preserves_density_units() -> None:
    losses = censored_exponential_nll(
        torch.tensor([math.log(0.5), math.log(0.5)]),
        torch.tensor([172_800.0, 172_800.0]),
        torch.tensor([True, False]),
        reduction="none",
        duration_scale=86_400.0,
    )

    assert torch.isclose(
        losses[0],
        torch.tensor(1.0 - math.log(0.5) + math.log(86_400.0)),
    )
    assert torch.isclose(losses[1], torch.tensor(1.0))


def test_time_normalization_guard_rejects_unscaled_or_nonfinite_durations() -> None:
    with pytest.raises(ValueError, match="time scale"):
        censored_exponential_nll(
            torch.zeros(1),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            duration_scale=0.0,
        )
    with pytest.raises(ValueError, match="normalized-time guard"):
        censored_exponential_nll(
            torch.zeros(1),
            torch.tensor([2_000_000.0]),
            torch.ones(1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="finite"):
        censored_exponential_nll(
            torch.zeros(1),
            torch.tensor([float("nan")]),
            torch.ones(1, dtype=torch.bool),
        )
