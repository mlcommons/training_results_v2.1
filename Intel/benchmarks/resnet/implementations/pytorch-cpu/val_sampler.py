import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

__all__ = ["DistributedValSampler", ]

T_co = TypeVar('T_co', covariant=True)

"""
We modify the DistributedSampler from torch.utils.data for distributed evaluation. 
In particular: 
  * indices are not shuffled and sequential ordering maintained. 
  * no extra samples are added - the last rank will see fewer samples if total 
    validation samples is not divisible by number of ranks.  

"""
class DistributedValSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None) -> None:
                 
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.samples_per_replica = math.ceil(len(self.dataset)/self.num_replicas)  # type: ignore[arg-type]
        self.total_size = len(self.dataset)   # type: ignore[arg-type]

        if self.rank == (self.num_replicas - 1):
            self.num_samples = self.total_size - (self.num_replicas-1)*self.samples_per_replica
        else:
            self.num_samples = self.samples_per_replica

    def __iter__(self) -> Iterator[T_co]:

        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # sequential assign indices for each rank   
        start_idx = self.rank*self.samples_per_replica   
        end_idx = (self.rank + 1)*self.samples_per_replica

        # subsample
        indices = indices[start_idx:end_idx] 
        assert len(indices) <= self.samples_per_replica

        return iter(indices)

    def __len__(self) -> int:

        if self.rank == (self.num_replicas - 1):
            num_samples = self.total_size - (self.num_replicas-1)*self.samples_per_replica
        else:
            num_samples = self.samples_per_replica

        return num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        pass 