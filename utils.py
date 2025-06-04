import json
from typing import Dict, Any
from datasets import load_dataset, DatasetDict

import config


def load_kt_dataset() -> DatasetDict:
    """
    Load and split the Kotlin dataset for training, evaluation, and testing.
    
    Returns:
        DatasetDict: A dictionary containing 'train', 'eval', and 'test' splits
    """
    try:
        ds_kt = load_dataset('json', data_files=config.KT_DS_PATH)['train'] \
            .rename_column('labels', 'labels') \
            .shard(num_shards=config.DATASET_SHARD_SIZE, index=config.DATASET_SHARD_INDEX)
        
        ds_kt = ds_kt.train_test_split(test_size=config.TEST_SIZE_RATIO)
        ds_kt_remaining = ds_kt['test'].train_test_split(test_size=config.EVAL_TEST_SPLIT_RATIO)
        
        ds_kt['eval'] = ds_kt_remaining['train']
        ds_kt['test'] = ds_kt_remaining['test']
        
        return ds_kt
    except Exception as e:
        raise RuntimeError(f"Failed to load Kotlin dataset: {e}")

