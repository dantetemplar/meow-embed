import torch

from meow_embed.server import resolve_rerank_activation_fn


def test_resolve_rerank_activation_fn_default() -> None:
    assert resolve_rerank_activation_fn(None) is None
    assert resolve_rerank_activation_fn("default") is None


def test_resolve_rerank_activation_fn_identity() -> None:
    fn = resolve_rerank_activation_fn("identity")
    assert isinstance(fn, torch.nn.Identity)
    assert fn(torch.tensor(1.5)).item() == 1.5


def test_resolve_rerank_activation_fn_sigmoid() -> None:
    fn = resolve_rerank_activation_fn("sigmoid")
    assert isinstance(fn, torch.nn.Sigmoid)
    assert 0.0 < fn(torch.tensor(0.0)).item() < 1.0
