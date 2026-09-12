"""TrainUNet: a dual-head (Actor-Critic) U-Net for the Railroad Tycoon AlphaZero-style agent.

The encoder downsamples the board, the decoder (with skip connections) upsamples it back out
to a per-cell policy heatmap (the Actor), and a separate Value head branches directly off the
bottleneck - the lowest-resolution, most compressed representation of the whole board - to
predict a single scalar win estimate (the Critic).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from railroad_env.game_state import GameState
from .encoding import BOARD_HEIGHT, BOARD_WIDTH

IN_CHANNELS = GameState.NUM_CHANNELS  # 28
POLICY_CHANNELS = 2  # 0 = PLACE heatmap, 1 = DISRUPT heatmap

# Two 2x2 max-pools take (20,30) -> (10,15) -> (5,7). 15 isn't evenly halvable (15 -> 7 with a
# remainder), so the decoder upsamples to explicit target sizes rather than a fixed scale_factor -
# otherwise 7*2=14 would mismatch the (10,15) skip connection's width.
def _pooled_size(size: int) -> int:
    return size // 2


STAGE1_HW = (BOARD_HEIGHT, BOARD_WIDTH)                                  # (20, 30)
STAGE2_HW = (_pooled_size(STAGE1_HW[0]), _pooled_size(STAGE1_HW[1]))     # (10, 15)
BOTTLENECK_HW = (_pooled_size(STAGE2_HW[0]), _pooled_size(STAGE2_HW[1])) # (5, 7)


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU, twice. The standard U-Net building block used at every
    encoder stage, the bottleneck, and every decoder stage."""

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


class TrainUNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        value_hidden: int = 256,
        dropout: float = 0.1,
        policy_channels: int = POLICY_CHANNELS,
        bounded_value: bool = True,
    ):
        """`bounded_value` puts a Tanh on the value head, so it predicts a normalized outcome in
        [-1, 1] - right for the AlphaZero-style path, where the target is a win/loss margin. PPO
        regresses onto discounted returns whose scale isn't bounded, so it passes False."""
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4  # 64, 128, 256

        # --- Encoder ---
        self.enc1 = ConvBlock(IN_CHANNELS, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck (also feeds the Value head) ---
        self.bottleneck = ConvBlock(c2, c3)

        # --- Value head: branches directly off the bottleneck ---
        bottleneck_flat = c3 * BOTTLENECK_HW[0] * BOTTLENECK_HW[1]
        value_layers = [
            nn.Flatten(),
            nn.Linear(bottleneck_flat, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(value_hidden, 1),
        ]
        if bounded_value:
            value_layers.append(nn.Tanh())
        self.value_head = nn.Sequential(*value_layers)

        # --- Decoder (Policy head), with skip connections from the encoder ---
        # A 1x1 conv reduces the upsampled tensor's channels to match its skip connection
        # before concatenation, keeping the resulting channel count (and therefore the
        # following ConvBlock's input size) fixed regardless of base_channels.
        self.up2_reduce = nn.Conv2d(c3, c2, kernel_size=1)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1_reduce = nn.Conv2d(c2, c1, kernel_size=1)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.policy_head = nn.Conv2d(c1, policy_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor = None):
        """
        x: (B, IN_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
        action_mask: optional (B, 2, BOARD_HEIGHT, BOARD_WIDTH) or (2, BOARD_HEIGHT, BOARD_WIDTH),
            1.0 = legal. Illegal cells get their pre-softmax logit set to -1e9 - the caller
            (MCTS, or the training loss) is responsible for the softmax itself; this method
            only ever returns raw (masked) logits.
        Returns: (policy_logits (B, 2, BOARD_HEIGHT, BOARD_WIDTH), value (B, 1))
        """
        skip1 = self.enc1(x)                    # (B, c1, 20, 30)
        skip2 = self.enc2(self.pool(skip1))      # (B, c2, 10, 15)
        bottleneck = self.bottleneck(self.pool(skip2))  # (B, c3, 5, 7)

        value = self.value_head(bottleneck)      # (B, 1); in [-1, 1] if bounded_value

        up2 = F.interpolate(bottleneck, size=STAGE2_HW, mode="bilinear", align_corners=False)
        up2 = self.up2_reduce(up2)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))  # (B, c2, 10, 15)

        up1 = F.interpolate(dec2, size=STAGE1_HW, mode="bilinear", align_corners=False)
        up1 = self.up1_reduce(up1)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))  # (B, c1, 20, 30)

        policy_logits = self.policy_head(dec1)   # (B, 2, 20, 30)

        if action_mask is not None:
            illegal = action_mask == 0
            # -1e9 (the literal "very negative" value) overflows float16's ~65504 max magnitude,
            # which this tensor becomes under torch.amp.autocast during mixed-precision training.
            # torch.finfo(dtype).min picks the most extreme representable negative value for
            # whatever dtype is actually in play (float32 or float16) - functionally identical
            # to -1e9 for softmax purposes (its exponential still underflows to exact 0.0), just
            # safe under autocast too.
            mask_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(illegal, mask_value)

        return policy_logits, value
