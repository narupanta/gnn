import torch
import torch.nn as nn

class Normalizer(nn.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, batch_size, feature_size, name, device, max_accumulations=10 ** 6, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
        self._name = name
        self._device = device
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon
        
        self._acc_count = 0
        self._num_accumulations = 0
        self._acc_sum = torch.zeros((batch_size, feature_size), dtype=torch.float, device = device)
        self._acc_sum_squared = torch.zeros((batch_size, feature_size), dtype=torch.float, device = device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data)
        return (batched_data - self._mean().unsqueeze(1)) / self._std_with_epsilon().unsqueeze(1)

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return torch.einsum('ij,ikj->ikj', self._std_with_epsilon(), normalized_batch_data) + self._mean().unsqueeze(1)

    def _accumulate(self, batched_data, node_num=None):
        """Function to perform the accumulation of the batch_data statistics."""
        data_sum = torch.sum(batched_data, dim=1)
        squared_data_sum = torch.sum(batched_data ** 2, dim=1)
        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += batched_data.shape[1]
        self._num_accumulations += 1

    def _mean(self):
        safe_count = max(self._acc_count, 1)
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = max(self._acc_count, 1)
        var = self._acc_sum_squared / safe_count - self._mean() ** 2
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + self._std_epsilon)
        return std

    def get_acc_sum(self):
        return self._acc_sum

# class Normalizer(nn.Module):
#     """Feature normalizer that accumulates statistics online."""

#     def __init__(self, size, name, max_accumulations=10 ** 6, std_epsilon=1e-8, ):
#         super(Normalizer, self).__init__()
#         self._name = name
#         self._max_accumulations = max_accumulations
#         self._std_epsilon = torch.tensor([std_epsilon], requires_grad=False)

#         self._acc_count = torch.zeros(1, dtype=torch.float32, requires_grad=False)
#         self._num_accumulations = torch.zeros(1, dtype=torch.float32, requires_grad=False)
#         self._acc_sum = torch.zeros(size, dtype=torch.float32, requires_grad=False)
#         self._acc_sum_squared = torch.zeros(size, dtype=torch.float32, requires_grad=False)

#     def forward(self, batched_data, node_num=None, accumulate=True):
#         """Normalizes input data and accumulates statistics."""
#         if accumulate and self._num_accumulations < self._max_accumulations:
#             # stop accumulating after a million updates, to prevent accuracy issues
#             self._accumulate(batched_data)
#         return (batched_data - self._mean()) / self._std_with_epsilon()

#     def inverse(self, normalized_batch_data):
#         """Inverse transformation of the normalizer."""
#         return normalized_batch_data * self._std_with_epsilon() + self._mean()

#     def _accumulate(self, batched_data, node_num=None):
#         """Function to perform the accumulation of the batch_data statistics."""
#         count = torch.tensor(batched_data.shape[0], dtype=torch.float32)

#         data_sum = torch.sum(batched_data, dim=0)
#         squared_data_sum = torch.sum(batched_data ** 2, dim=0)
#         self._acc_sum = self._acc_sum.add(data_sum)
#         self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
#         self._acc_count = self._acc_count.add(count)
#         self._num_accumulations = self._num_accumulations.add(1.)

#     def _mean(self):
#         safe_count = max(self._acc_count, 1)
#         return self._acc_sum / safe_count

#     def _std_with_epsilon(self):

#         safe_count = max(self._acc_count, 1)
#         std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
#         return torch.clamp(std, min=self._std_epsilon)
    
#     def get_acc_sum(self):
#         return self._acc_sum