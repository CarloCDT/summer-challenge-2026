"""Quantization-aware training: make the student learn weights that survive int4.

The shipped agent is int4. Measured over 250 head-to-head games, the same student scores 0.570
at full precision and 0.500 baked at int4 - the largest remaining loss between what we train and
what we send, now that distillation itself has caught up with its teacher (0.570 vs 0.600).
Nothing about the architecture or the distillation fixes that, because the training never sees
the rounding. This does: the forward pass uses rounded weights, the backward pass updates the
full-precision ones (a straight-through estimator), so the student learns a weight configuration
that is still good AFTER rounding.

WHY QUANTIZING THE RAW CONV WEIGHT IS THE RIGHT THING, even though bake_agent.py quantizes the
BatchNorm-FOLDED weight: folding multiplies each output channel by gamma/sqrt(var+eps), and this
quantizer picks its scale per output channel, so a per-channel rescale passes straight through it
- quantize-then-fold and fold-then-quantize give the same numbers. Verified against
bake_agent.fold_batchnorm on a real checkpoint: worst relative difference 4.4e-8 on the layers
that have BatchNorm, and exactly 0.0 on the three that do not. So there is no need to fold during
training, and no approximation in not doing so.

Biases are deliberately left alone - bake_agent keeps them float32.
"""
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

#: int4 stores 15 levels, -7..+7 (see bake_agent.quantize).
INT4_QMAX = 7
INT8_QMAX = 127


class FakeQuantPerChannel(nn.Module):
    """Symmetric per-output-channel round-trip, with a straight-through gradient.

    Forward returns the quantized weight; backward passes the gradient to the underlying
    full-precision weight unchanged. The scale is detached: it is a function of the weight, but
    differentiating through `max` would put the whole channel's gradient on one element.
    """

    def __init__(self, qmax: int = INT4_QMAX):
        super().__init__()
        self.qmax = int(qmax)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, weight.dim()))
        scale = weight.detach().abs().amax(dim=dims, keepdim=True) / self.qmax
        scale = scale.clamp_min(torch.finfo(weight.dtype).tiny)
        q = torch.round(weight / scale).clamp_(-self.qmax, self.qmax) * scale
        return weight + (q - weight).detach()


def attach_fake_quant(model: nn.Module, qmax: int = INT4_QMAX) -> int:
    """Fake-quantize every Conv2d weight in `model`. Returns how many were wrapped.

    Conv2d only, which is exactly the set bake_agent.LAYERS carries: the value head is Linear
    and is dropped at bake time, so quantizing it would cost accuracy for nothing.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and not parametrize.is_parametrized(module, "weight"):
            parametrize.register_parametrization(module, "weight", FakeQuantPerChannel(qmax))
            count += 1
    return count


def plain_state_dict(model: nn.Module, quantized: bool = False) -> dict:
    """A state dict with ordinary `...weight` keys, loadable by an unparametrized model.

    Parametrization renames `layer.weight` to `layer.parametrizations.weight.original`, which a
    plain RailroadUNet cannot load - and every consumer here (checkpoint files, the rollout
    workers, evaluate_all_bosses) builds a plain model.

    `quantized=False` returns the full-precision weights: that is what belongs in a checkpoint,
    because bake_agent.py applies the identical quantization itself and round-tripping twice
    would only throw away information the next training run could use.
    `quantized=True` returns the rounded weights - use it wherever the number should reflect what
    actually ships, i.e. rollout and evaluation.
    """
    out = {}
    for key, value in model.state_dict().items():
        out[key.replace(".parametrizations.weight.original", ".weight")] = value.detach().clone()
    if quantized:
        for name, module in model.named_modules():
            if parametrize.is_parametrized(module, "weight"):
                out[f"{name}.weight"] = module.weight.detach().clone()
    return out
