from tqdm import tqdm
import numpy as np
from data_parser import KT_DS_PATH


def load_kt_dataset():
    ds_kt = load_dataset('json', data_files=KT_DS_PATH))['train']\
        .rename_column('target', 'labels')\
        .shard(num_shards=40, index=0)

    ds_kt = ds_kt.train_test_split(test_size=0.3)

    ds_kt_remaining = ds_kt['test'].train_test_split(test_size=0.5)

    ds_kt['eval'] = ds_kt_remaining['train']
    ds_kt['test'] = ds_kt_remaining['test']
    return ds_kt

