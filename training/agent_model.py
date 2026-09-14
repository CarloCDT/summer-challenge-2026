"""RailroadUNet: the dual-head U-Net used by the PPO agents.

    conv in  ->  enc1 ---------------------------- concat -> dec1 -> conv out (policy heatmaps)
                   |                                  |
                 pool                             upsample
                   |                                  |
                 enc2 ------------- concat -------> dec2
                   |                  |
                 pool             upsample
                   |                  |
                 bottleneck -----------
                   |
                 critic (value)

Two downsampling stages and two upsampling stages, with a plain convolution before the encoder
and after the decoder. The critic reads the bottleneck - the most compressed view of the board -
through a small spatial pool, which keeps the value head from dominating the parameter count the
way a full flatten would.

The value head is unbounded: PPO regresses onto discounted returns, whose scale isn't confined to
[-1, 1] the way an AlphaZero win-margin target is.

Policy channels differ by game:
  * lite  - 1 channel:  PLACE
  * full  - 3 channels: PLACE, DISRUPT, SKIP_DISRUPT (see training/simulator.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import BOARD_HEIGHT, BOARD_WIDTH


def _pooled_size(size: int) -> int:
    return size // 2


# Two 2x2 max-pools take (20,30) -> (10,15) -> (5,7). 15 doesn't halve evenly, so the decoder
# upsamples to explicit target sizes rather than a fixed scale_factor - a naive 2x would turn 7
# back into 14 and mismatch the (10,15) skip connection.
STAGE1_HW = (BOARD_HEIGHT, BOARD_WIDTH)                                   # (20, 30)
STAGE2_HW = (_pooled_size(STAGE1_HW[0]), _pooled_size(STAGE1_HW[1]))      # (10, 15)
BOTTLENECK_HW = (_pooled_size(STAGE2_HW[0]), _pooled_size(STAGE2_HW[1]))  # (5, 7)

# The critic pools the bottleneck down to this before its MLP.
VALUE_POOL_HW = (2, 3)


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU, twice."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RailroadUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        policy_channels: int,
        base_channels: int = 32,
        value_hidden: int = 128,
        dropout: float = 0.1,
        channels=None,
    ):
        """`channels` gives the three stage widths explicitly as (c1, c2, c3); leave it None for
        the classic doubling progression (c, 2c, 4c) derived from `base_channels`.

        Why the override exists: doubling concentrates the weights in the coarsest stages, which
        run on 5x7 and 10x15 grids - at (64,128,256) the bottleneck and dec2 hold 61% of the
        parameters. The policy output is a per-cell heatmap over the full 20x30 board, so under a
        hard size budget (CodinGame caps source at 100k chars) a flatter progression buys far more
        capacity where the decision is actually made: (34,34,36) and (16,32,64) both bake under
        that cap at int4, but the flat one has 3.9x more parameters running at full resolution.
        See training/configs/distill.yaml for the measured size table."""
        super().__init__()
        # Recorded so forward() can slice a wider observation down to what this net was built
        # for: channels are only ever APPENDED, so a 28-channel checkpoint reads the same first 28.
        self.in_channels = int(in_channels)
        if channels is not None:
            c1, c2, c3 = (int(c) for c in channels)
        else:
            c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # --- convolution before ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # --- 2 down ---
        self.enc1 = ConvBlock(c1, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c2, c3)

        # --- critic, off the bottleneck ---
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(VALUE_POOL_HW),
            nn.Flatten(),
            nn.Linear(c3 * VALUE_POOL_HW[0] * VALUE_POOL_HW[1], value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(value_hidden, 1),
        )

        # --- 2 up, with skip connections ---
        # A 1x1 conv trims the upsampled tensor's channels to match its skip connection before
        # concatenation, so the following ConvBlock's input size doesn't depend on base_channels.
        self.up2_reduce = nn.Conv2d(c3, c2, kernel_size=1)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1_reduce = nn.Conv2d(c2, c1, kernel_size=1)
        self.dec1 = ConvBlock(c1 * 2, c1)

        # --- convolution after ---
        self.head = nn.Conv2d(c1, policy_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor):
        """Stem and encoder -> (skip1, skip2, bottleneck).

        An input with MORE channels than this net was built for is sliced to its first
        `in_channels`. Observation channels are only ever appended, so a checkpoint trained on the
        28-channel layout reads exactly the planes it learned from a 38-channel observation."""
        if x.shape[1] != self.in_channels:
            if x.shape[1] < self.in_channels:
                raise ValueError(
                    f"RailroadUNet was built for {self.in_channels} input channels, got {x.shape[1]}"
                )
            x = x[:, :self.in_channels]
        skip1 = self.enc1(self.stem(x))                  # (B, c1, 20, 30)
        skip2 = self.enc2(self.pool(skip1))              # (B, c2, 10, 15)
        bottleneck = self.bottleneck(self.pool(skip2))   # (B, c3, 5, 7)
        return skip1, skip2, bottleneck

    def value_only(self, x: torch.Tensor) -> torch.Tensor:
        """The critic alone: stem, encoder, value head - no decoder, no policy head. This is how a
        `critic: separate` network is read (train_ppo.py), at roughly half a full forward pass."""
        return self.value_head(self.encode(x)[2])

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor = None):
        """
        x: (B, C >= in_channels, BOARD_HEIGHT, BOARD_WIDTH) - extra trailing channels are ignored
        action_mask: optional (B, policy_channels, BOARD_HEIGHT, BOARD_WIDTH), 1.0 = legal.
            Illegal cells get their pre-softmax logit driven to the dtype's minimum; the caller
            owns the softmax, this only ever returns raw (masked) logits.
        Returns: (policy_logits (B, policy_channels, H, W), value (B, 1))
        """
        skip1, skip2, bottleneck = self.encode(x)

        value = self.value_head(bottleneck)              # (B, 1), unbounded

        up2 = F.interpolate(bottleneck, size=STAGE2_HW, mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([self.up2_reduce(up2), skip2], dim=1))

        up1 = F.interpolate(dec2, size=STAGE1_HW, mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([self.up1_reduce(up1), skip1], dim=1))

        policy_logits = self.head(dec1)

        if action_mask is not None:
            # finfo.min rather than a literal -1e9: under autocast this tensor can be float16,
            # whose max magnitude (~65504) -1e9 would overflow. Either way the exponential
            # underflows to exactly 0.0, so masked cells contribute nothing to the softmax.
            policy_logits = policy_logits.masked_fill(
                action_mask == 0, torch.finfo(policy_logits.dtype).min
            )

        return policy_logits, value


def expand_input_channels(state_dict: dict, in_channels: int):
    """Widen a checkpoint's stem to `in_channels`, giving every new input channel ZERO weights.

    Returns (state_dict, the checkpoint's original in_channels). The stem is the only layer that
    sees the raw observation and a zero-weight channel contributes exactly nothing to it, so the
    widened network computes the identical function - BatchNorm statistics included - until
    training moves those weights. This is what lets a 28-channel teacher warm-start a 38-channel
    run with no loss of play."""
    weight = state_dict["stem.0.weight"]
    old = weight.shape[1]
    if old == in_channels:
        return state_dict, old
    if old > in_channels:
        raise ValueError(f"checkpoint stem reads {old} channels, more than the {in_channels} requested")
    pad = torch.zeros(weight.shape[0], in_channels - old, *weight.shape[2:],
                      dtype=weight.dtype, device=weight.device)
    widened = dict(state_dict)
    widened["stem.0.weight"] = torch.cat([weight, pad], dim=1)
    return widened, old


def rescale_value_head(state_dict: dict, factor: float) -> dict:
    """Multiply the critic's output by `factor`, exactly, by scaling its final Linear. A warm start
    that changes value_scale uses it so V(s) in return units stays what the checkpoint predicted."""
    scaled = dict(state_dict)
    for key in ("value_head.5.weight", "value_head.5.bias"):
        scaled[key] = state_dict[key] * factor
    return scaled
