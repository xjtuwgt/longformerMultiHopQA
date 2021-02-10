import torch
import numpy as np
import json
import sys
import operator
from functools import reduce

from tqdm import tqdm
from collections import Counter

assert len(sys.argv) == 4
raw_data = json.load(open(sys.argv[1], 'r'))
longformer_rank_data = json.load(open(sys.argv[2], 'r'))
hgn_rank_data = json.load(open(sys.argv[3], 'r'))
output_file = sys.argv[4]

def data_combined(gold_titles, long_titles, hgn_titles):

    return

for case in tqdm(raw_data):
    guid = case['_id']
    gold_titles = list(set([_[0]  for _ in case['supporting_facts']]))
    long_titles = reduce(operator.concat, longformer_rank_data[guid])
    hgn_titles = reduce(operator.concat, hgn_rank_data[guid])



