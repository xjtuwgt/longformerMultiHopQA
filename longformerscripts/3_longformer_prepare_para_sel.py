from __future__ import absolute_import, division, print_function
import sys
from pandas import DataFrame
import pandas as pd
from time import time
from longformerDataUtils.RetrievalOfflineProcess import Hotpot_Retrieval_Train_Dev_Data_Preprocess
from longformerscripts.longformerUtils import get_hotpotqa_longformer_tokenizer

########################################################################################################################
# data preprocess for longformer (long sequence based retrieval)
########################################################################################################################
def loadJSONData(json_fileName)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(json_fileName, orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

input_file = sys.argv[1]
output_file = sys.argv[2]

longformer_tokenizer = get_hotpotqa_longformer_tokenizer()
data_frame = loadJSONData(json_fileName=input_file)
_, combined_data_res, _ = Hotpot_Retrieval_Train_Dev_Data_Preprocess(data=data_frame, tokenizer=longformer_tokenizer)
combined_data_res.to_json(output_file)