import torch

from braid.model.memory import RoleAwareNodeMemory
from braid.model.ties import pool_tie_groups


def test_tie_pooling_is_invariant_to_serialization_order() -> None:
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]]])
    groups = torch.tensor([[4, 4, 5]])
    pooled_a, mask_a, ids_a = pool_tie_groups(values, groups)
    pooled_b, mask_b, ids_b = pool_tie_groups(values[:, [1, 0, 2]], groups)
    torch.testing.assert_close(pooled_a, pooled_b)
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(ids_a, ids_b)


def test_node_memory_is_role_sensitive() -> None:
    torch.manual_seed(4)
    module = RoleAwareNodeMemory(8)
    initial = module.initial_state(torch.zeros(1, 2, 8))
    events = torch.ones(1, 1, 8)
    nodes = torch.tensor([[[0]]])
    role_a = torch.zeros(1, 1, 1, 8)
    role_b = torch.ones(1, 1, 1, 8)
    memory_a = module.update_tie_group(initial, events, nodes, role_a)
    memory_b = module.update_tie_group(initial, events, nodes, role_b)
    assert not torch.allclose(memory_a[:, 0], memory_b[:, 0])
    torch.testing.assert_close(memory_a[:, 1], memory_b[:, 1])
