"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

from fairchem.core.common import distutils, gp_utils


def simple_batch_sampler(
    dataset: Dataset,
    num_replicas: int,
    rank: int,
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
):
    """
    Simple batch sampler using PyTorch's built-in DistributedSampler and BatchSampler.
    Just randomly picks batches as usual - no special atom-based balancing.
    """
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    return BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)


def get_dataloader(
    dataset: Dataset, batch_sampler_fn: callable, collate_fn: callable, num_workers
) -> DataLoader:
    if gp_utils.initialized():
        num_replicas = gp_utils.get_dp_world_size()
        rank = gp_utils.get_dp_rank()
    else:
        num_replicas = distutils.get_world_size()
        rank = distutils.get_rank()

    logging.info(f"get_dataloader::Calling batch_sampler_fn={batch_sampler_fn}...")
    batch_sampler = batch_sampler_fn(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank,
    )
    logging.info("get_dataloader::Calling Dataloader...")
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=batch_sampler,
    )
    logging.info("get_dataloader::Done!")
    return dataloader
