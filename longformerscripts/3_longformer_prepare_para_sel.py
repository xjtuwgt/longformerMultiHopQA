from __future__ import absolute_import, division, print_function
import os
import sys
from longformerDataUtils.ioutils import loadJSONData
from longformerDataUtils.RetrievalOfflineProcess import Hotpot_Retrieval_Train_Dev_Data_Preprocess
from longformerscripts.longformerUtils import get_hotpotqa_longformer_tokenizer

data_path = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

longformer_tokenizer = get_hotpotqa_longformer_tokenizer()
data_frame = loadJSONData(PATH=data_path, json_fileName=input_file)
all_data, combined_data_res, ind_norm_dev_data_res = Hotpot_Retrieval_Train_Dev_Data_Preprocess(data=data_frame,
                                                                                                tokenizer=longformer_tokenizer)
combined_data_res.to_json(os.path.join(data_path, output_file))