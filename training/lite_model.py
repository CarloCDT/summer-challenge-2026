"""LiteUNet: RailroadUNet preset for the lite game (16 input channels, PLACE-only policy).

The architecture itself lives in `training/agent_model.py` - this is just the lite configuration
of it, kept as its own name because the lite checkpoints and notebook refer to it.
"""
from railroad_lite_env import NUM_LITE_CHANNELS

from .agent_model import BOTTLENECK_HW, STAGE1_HW, STAGE2_HW, VALUE_POOL_HW, ConvBlock, RailroadUNet

IN_CHANNELS = NUM_LITE_CHANNELS  # 16
POLICY_CHANNELS = 1              # PLACE only - the lite game has no DISRUPT

__all__ = [
    "LiteUNet",
    "IN_CHANNELS",
    "POLICY_CHANNELS",
    "STAGE1_HW",
    "STAGE2_HW",
    "BOTTLENECK_HW",
    "VALUE_POOL_HW",
    "ConvBlock",
]


class LiteUNet(RailroadUNet):
    def __init__(
        self,
        base_channels: int = 32,
        value_hidden: int = 128,
        dropout: float = 0.1,
        policy_channels: int = POLICY_CHANNELS,
        in_channels: int = IN_CHANNELS,
    ):
        super().__init__(
            in_channels=in_channels,
            policy_channels=policy_channels,
            base_channels=base_channels,
            value_hidden=value_hidden,
            dropout=dropout,
        )
