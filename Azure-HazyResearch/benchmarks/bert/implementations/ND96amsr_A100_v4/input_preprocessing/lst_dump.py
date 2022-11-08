import argparse
import logging
import json
import glob
from collections import Counter

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(
    description="Training data sharding for BERT.")
parser.add_argument(
    '--input_lst',
    type=str,
    default='./',
    help='Input lst_file path')
args = parser.parse_args()

input_files = sorted(glob.glob(args.input_lst + '/shard_list_0*.lst', recursive=False))
all_ind=[]
for input_file in input_files:
    with open(input_file, 'r') as f:
        ind=json.load(f)
    all_ind += ind
    print(f"shard size: {len(ind)}")


tr = list(zip(*all_ind))
shards=Counter(tr[0])
for shard, num in sorted(shards.items(), key=lambda pair: pair[0], reverse=True):
    assert num==36279, f"failed {shard}, {num}"
indices=Counter(tr[1])
for idx, num in sorted(indices.items(), key=lambda pair: pair[0], reverse=True):
    assert num==4320 , f"failed {idx}, {num}"

