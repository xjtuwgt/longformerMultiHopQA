import gzip
import pickle
import json
import torch
import numpy as np
import argparse
import os
from sklearn.metrics import confusion_matrix
from eval.hotpot_evaluate_v1 import normalize_answer

from os.path import join
from collections import Counter
from longformerscripts.longformerIREvaluation import recall_computation

from model_envs import MODEL_CLASSES
from jd_mhqa.jd_data_processing import Example, InputFeatures, get_cached_filename
from UnifiedQAExample.UnifiedQAModel import unified_qa_prediction, unifiedqa_model_loader
from utils.gpu_utils import gpu_setting

def data_collection(raw_data, features, tokenizer):
    gold_answer_dict = {}
    decoded_query_dict = {}
    decoded_context_trim512_dict = {}
    count = 0
    for row in raw_data:
        qid = row['_id']
        gold_answer = row['answer']
        gold_answer_dict[qid] = gold_answer
        ################################################################################################################
        feature = features[qid]
        feature_dict = vars(feature)
        doc_input_ids = feature_dict['doc_input_ids']
        doc_tokens = feature_dict['doc_tokens']
        # print('Document length: token = {}, id = {}'.format(len(doc_tokens), len(doc_input_ids)))
        decoded_context_text = tokenizer.decode(doc_input_ids, skip_special_tokens=True)
        decoded_context_trim512_dict[qid] = decoded_context_text
        assert len(doc_input_ids) == 512
        query_tokens = feature_dict['query_tokens']
        query_input_ids = feature_dict['query_input_ids']
        # print('Query length: token = {}, id = {}'.format(len(query_tokens), len(query_input_ids)))
        decoded_query_text = tokenizer.decode(query_input_ids, skip_special_tokens=True)
        decoded_query_dict[qid] = decoded_query_text
        print('{}\n{}'.format(decoded_query_text, decoded_context_text))
        print('*' * 75)
        count = count + 1
        if count % 500 == 0:
            print('Processing {} records'.format(count))

    processed_data_dict = {'answer': gold_answer_dict, 'query': decoded_query_dict, 'context': decoded_context_trim512_dict}
    return processed_data_dict


def model_evaluation(raw_data, features, tokenizer, unified_qa_model, unified_qa_tokenizer):
    predicted_answer_list = []
    gold_answer_list = []
    for row in raw_data:
        qid = row['_id']
        answer = row['answer']
        gold_answer_list.append(normalize_answer(answer))
        ################################################################################################################
        feature = features[qid]
        feature_dict = vars(feature)
        doc_input_ids = feature_dict['doc_input_ids']
        doc_tokens = feature_dict['doc_tokens']
        print('Document length: token = {}, id = {}'.format(len(doc_tokens), len(doc_input_ids)))
        decoded_context_text = tokenizer.decode(doc_input_ids, skip_special_tokens=True)
        assert len(doc_input_ids) == 512
        query_tokens = feature_dict['query_tokens']
        query_input_ids = feature_dict['query_input_ids']
        print('Query length: token = {}, id = {}'.format(len(query_tokens), len(query_input_ids)))
        decoded_query_text = tokenizer.decode(query_input_ids, skip_special_tokens=True)
        # if answer not in ['yes', 'no']:
        #     unified_answer = unified_qa_prediction(model=unified_qa_model, tokenizer=unified_qa_tokenizer, question=decoded_query_text, context=decoded_context_text)
        #     predicted_answer_list.append(normalize_answer(unified_answer))

def load_data_from_disk(args):
    cached_examples_file = os.path.join(args.input_dir,
                                        get_cached_filename('examples', args))
    cached_features_file = os.path.join(args.input_dir,
                                        get_cached_filename('features', args))
    cached_graphs_file = os.path.join(args.input_dir,
                                      get_cached_filename('graphs', args))

    examples = pickle.load(gzip.open(cached_examples_file, 'rb'))
    features = pickle.load(gzip.open(cached_features_file, 'rb'))
    graph_dict = pickle.load(gzip.open(cached_graphs_file, 'rb'))
    example_dict = {example.qas_id: example for example in examples}
    feature_dict = {feature.qas_id: feature for feature in features}
    with open(args.raw_data, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)
    print('Loading raw data from: {}'.format(args.raw_data))
    print("Loading examples from: {}".format(cached_examples_file))
    print("Loading features from: {}".format(cached_features_file))
    print("Loading graphs from: {}".format(cached_graphs_file))
    return raw_data, example_dict, feature_dict, graph_dict

def device_setting(args):
    if torch.cuda.is_available():
        free_gpu_ids, used_memory = gpu_setting(num_gpu=args.gpus)
        print('{} gpus with used memory = {}, gpu ids = {}'.format(len(free_gpu_ids), used_memory, free_gpu_ids))
        if args.gpus > 0:
            gpu_ids = free_gpu_ids
            device = torch.device("cuda:%d" % gpu_ids[0])
            print('Single GPU setting')
        else:
            device = torch.device("cpu")
            print('Single cpu setting')
    else:
        device = torch.device("cpu")
        print('Single cpu setting')
    return device

if __name__ == '__main__':
    t5_small_model_name = "allenai/unifiedqa-t5-small"  # you can specify the model size here

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True, help='define output directory')
    parser.add_argument("--output_dir", type=str, required=True, help='define output directory')
    parser.add_argument("--pred_dir", type=str, required=True, help='define output directory')
    parser.add_argument("--graph_id", type=str, default="1", help='define output directory')

    # Other parameters
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default='', type=str, required=True,
                        help="Path to pre-trained model")

    parser.add_argument('--unified_qa_model_name_or_path', default='',
                        type=str, required=True,
                        help="Path to pre-trained model")

    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    orig_tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    # ##################################################################################################################
    raw_data, example_dict, feature_dict, graph_dict = load_data_from_disk(args=args)
    processed_data = data_collection(raw_data=raw_data, features=feature_dict, tokenizer=orig_tokenizer)
    with open(os.path.join(args.pred_dir, args.model_type, 'preprocessed_data.json'), 'w') as fp:
        json.dump(processed_data, fp)
    # ##################################################################################################################

    # # ##################################################################################################################
    # unified_qa_model, unified_qa_tokenizer = unifiedqa_model_loader(model_name=args.unified_qa_model_name_or_path)
    # # ##################################################################################################################
    # raw_data, example_dict, feature_dict, graph_dict = load_data_from_disk(args=args)
    # ##################################################################################################################
    # device = device_setting(args=args)
    # unified_qa_model = unified_qa_model.to(device)
    # model_evaluation(raw_data=raw_data, features=feature_dict, tokenizer=orig_tokenizer, unified_qa_tokenizer=unified_qa_tokenizer, unified_qa_model=unified_qa_model)

    # data_analysis(raw_data, example_dict, feature_dict, tokenizer, use_ent_ans=False)
    # metrics = hotpot_eval(pred_file, args.raw_data)
    # for key, val in metrics.items():
    #     print("{} = {}".format(key, val))
