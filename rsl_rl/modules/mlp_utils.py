"""Shared MLP construction for policy networks."""

import torch.nn as nn


def build_mlp(input_dim, hidden_dims, output_dim, activation, final_activation=False):
    """Build a sequential MLP.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer sizes.
        output_dim: Output dimension; if None, last hidden layer is the output (no final linear).
        activation: Module class or instance used for hidden activations (e.g. nn.ELU).
        final_activation: If True and output_dim is set, apply activation after last linear.
    """
    layers = []
    prev_dim = input_dim
    act_cls = activation if isinstance(activation, type) else type(activation)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_cls())
        prev_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, output_dim))
        if final_activation:
            layers.append(act_cls())
    elif not final_activation and layers:
        layers.pop()
    return nn.Sequential(*layers)
