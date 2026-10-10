"""Small correctness checks for the ordinary hard-code adversarial control."""

import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.adversary import (
    HardCodeCritics,
    balanced_bce,
)


def test_balanced_bce_uses_equal_group_mass():
    logits = torch.tensor([0.0, 0.0, 0.0, 4.0])
    sex = torch.tensor([0, 0, 0, 1])
    target = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:3], sex[:3].float()) +
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits[3:], sex[3:].float())) / 2
    assert torch.allclose(balanced_bce(logits, sex), target)


def test_mapper_loss_passes_gradient_to_actual_hard_bits_and_not_critics():
    torch.manual_seed(17)
    critics = HardCodeCritics()
    p1 = torch.tensor([[0.2] * 10, [0.8] * 10, [0.7] * 10, [0.3] * 10], requires_grad=True)
    p2 = 1 - p1
    bits1 = p1 + ((p1 >= 0.5).float() - p1).detach()
    bits2 = p2 + ((p2 >= 0.5).float() - p2).detach()
    sex = torch.tensor([0, 1, 1, 0])
    assert torch.equal(bits1.detach(), (p1.detach() >= 0.5).float())
    assert torch.equal(bits2.detach(), (p2.detach() >= 0.5).float())
    loss = critics.mapper_loss(bits1, bits2, sex, coefficient=1.0)
    loss.backward()
    assert p1.grad is not None and torch.isfinite(p1.grad).all()
    assert p1.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in critics.parameters())


def test_critic_step_uses_hard_bits_and_fits_pair_and_local_views():
    torch.manual_seed(19)
    critics = HardCodeCritics()
    optimizer = torch.optim.Adam(critics.parameters(), lr=0.01)
    bit1 = torch.tensor([[0.] * 10, [1.] * 10, [0.] * 10, [1.] * 10])
    bit2 = 1 - bit1
    sex = torch.tensor([0, 1, 0, 1])
    before = [parameter.detach().clone() for parameter in critics.parameters()]
    receipt = critics.critic_step(bit1, bit2, sex, optimizer)
    assert set(receipt) == {"local_1", "local_2", "pair"}
    assert all(value >= 0 for value in receipt.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, critics.parameters()))
