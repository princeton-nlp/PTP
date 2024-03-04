from transformers.utils import logging
import torch
from torch.utils.data import Dataset
import numpy as np
from streaming import LocalDataset, StreamingDataset, Stream
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os

logger = logging.get_logger(__name__)

class MDSDataset(StreamingDataset):

    def __init__(self, block_size=None, return_key="tokens", **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        if block_size is not None:
            logger.warning("block_size set in MDSDataset, which means the input might be truncated")
        self.return_key = return_key


    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        tokens = np.frombuffer(item["tokens"], np.uint16).astype(np.int64)
        if self.block_size is not None:
            tokens = tokens[:self.block_size]
        return {self.return_key: tokens}


redpajama_domains_and_proportions = {
    "arxiv": 0.025, 
    "book": 0.045, 
    "c4-rp": 0.15, 
    "cc": 0.67,
    "github": 0.045, 
    "stackexchange": 0.02, 
    "wiki": 0.045
}

def get_multiple_domain_dataset(
    root_dir,
    shuffle,
    domains_and_proportions=redpajama_domains_and_proportions,
    remote=False,
    block_size=None,
):
    logger.warning("Loading multiple domain dataset via MosaicML streaming.")
    logger.warning("***** Streaming dataset *****")
    logger.warning(f"Root dir: {root_dir}")
    logger.warning(f"Shuffle: {shuffle}")
    logger.warning(f"Domains: {domains_and_proportions}")
    logger.warning(f"Remote: {remote}")
    logger.warning(f"Block size: {block_size}")

    if remote:
        streams = [
            Stream(remote=root_dir+domain, proportion=domains_and_proportions[domain])
            for domain in domains_and_proportions
        ]
    else:
        streams = [
            Stream(local=os.path.join(root_dir, domain), proportion=domains_and_proportions[domain])
            for domain in domains_and_proportions
        ]
    
    dataset = MDSDataset(
        block_size=block_size,
        streams=streams,
        shuffle=shuffle,
    )

    return dataset