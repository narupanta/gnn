import torch
import torch.nn as nn

class Normalizer(nn.Module):
    """Feature normalizer for streaming (N, F) data (nodes or edges)."""

    def __init__(self, feature_size, device, max_accumulations=10**6, std_epsilon=1e-8):
        super().__init__()
        self._device = device
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon

        # Accumulators
        self._acc_count = 0
        self._num_accumulations = 0
        self._acc_sum = torch.zeros(feature_size, dtype=torch.float, device=device)
        self._acc_sum_squared = torch.zeros(feature_size, dtype=torch.float, device=device)

    def forward(self, data, accumulate=True):
        """
        Normalize input data of shape (N, F).
        """
        if accumulate and self._num_accumulations < self._max_accumulations:
            self._accumulate(data)

        return (data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_data):
        """
        Inverse normalization.
        """
        return normalized_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, data):
        """
        Update running statistics from data of shape (N, F).
        """
        self._acc_sum += data.sum(dim=0)
        self._acc_sum_squared += (data ** 2).sum(dim=0)
        self._acc_count += data.shape[0]
        self._num_accumulations += 1

    def _mean(self):
        safe_count = max(self._acc_count, 1)
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = max(self._acc_count, 1)
        mean = self._mean()
        var = self._acc_sum_squared / safe_count - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return std + self._std_epsilon

    def get_stats(self):
        return {
            "mean": self._mean().detach().cpu().numpy(),
            "std": self._std_with_epsilon().detach().cpu().numpy(),
            "count": self._acc_count
        }
