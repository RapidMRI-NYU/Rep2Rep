import torch


def make_pair(repetitions, sigma_map):
    """Pair independent repetitions 0 and 1; tensors use [N, C, H, W]."""
    if repetitions.ndim != 4 or repetitions.shape[0] < 2:
        raise ValueError("Expected at least two repetitions shaped [N, C, H, W]")
    if sigma_map.shape[0] == repetitions.shape[0]:
        sigma_map = sigma_map[:1]
    return repetitions[:1], repetitions[1:2], sigma_map
