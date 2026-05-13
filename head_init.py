"""
head_init.py — Xavier + small-scale initialization for the 100-class head.

Xavier uniform preserves gradient variance across layers, which is preferred
over Kaiming when the activation before the head is not ReLU. The small weight
scale (0.1x) prevents large initial logits that would produce a flat softmax
and a high initial loss — giving the ZO optimizer a better-shaped loss surface
to navigate from the first step.
"""

import torch
import torch.nn as nn
import math


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the 100-class CIFAR100 classification head.

    Strategy:
        - Xavier uniform weights scaled by 0.1 for a conservative start.
          Large initial weights → near-uniform softmax → high cross-entropy
          loss plateau that ZO methods struggle to escape.
        - Bias initialized to log(1/C) = log(1/100) ≈ -4.6 to match the
          uniform class prior, giving a well-calibrated starting point.

    Args:
        layer: The nn.Linear layer to initialize in-place.
    """
    nn.init.xavier_uniform_(layer.weight)
    layer.weight.data.mul_(0.1)  # conservative scale

    # Bias = log(uniform prior) for calibrated initial predictions
    num_classes = layer.out_features
    nn.init.constant_(layer.bias, math.log(1.0 / num_classes))
