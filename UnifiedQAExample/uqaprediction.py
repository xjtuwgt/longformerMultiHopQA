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
from eval.hotpot_evaluate_v1 import update_answer

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
        # print('{}\n{}'.format(decoded_query_text, decoded_context_text))
        # print('*' * 75)
        count = count + 1
        if count % 500 == 0:
            print('Processing {} records'.format(count))

    processed_data_dict = {'answer': gold_answer_dict, 'query': decoded_query_dict, 'context': decoded_context_trim512_dict}
    return processed_data_dict

def run_model(model, tokenizer, input_string, device, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = input_ids.to(device)
    if input_ids.shape[1] > 380:
        input_ids = input_ids[...,:380]
    print(input_ids.shape)
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

def unified_qa_evaluation(model, tokenizer, raw_data, pre_data, device):
    row_count = 0
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    predicted_answers = {}
    for row in raw_data:
        qid = row['_id']
        answer = row['answer']
        query = pre_data['query'][qid]
        context = pre_data['context'][qid]
        question_len = len(query)
        question = context[:question_len]
        # print('{}\n{}'.format(query, question))
        context_decoded = context[question_len:]
        # print('{}\n{}'.format(context_decoded, context))
        # print('*' * 75)
        # print('{}\n{}'.format(query, context))
        unified_qa_input = question + '\n' + context_decoded
        uni_answer = run_model(model, tokenizer, unified_qa_input, device)
        print('{}-th answer: {} | {}'.format(row_count + 1, answer, uni_answer[0]))
        em, prec, rec = update_answer(metrics=metrics, prediction=uni_answer[0], gold=answer)
        print('{}-th metric: {} | {} | {}'.format(row_count + 1, em, prec, rec))
        predicted_answers[qid] = uni_answer[0]
        print('*' * 75)
        row_count = row_count + 1
        if row_count % 1000 == 0:
            for key, value in metrics.items():
                print('{}:{}'.format(key, value * 1.0 / row_count))
    print(row_count)
    for key, value in metrics.items():
        print('{}:{}'.format(key, value*1.0/row_count))
    return predicted_answers

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
    unified_qa_model, unified_qa_tokenizer = unifiedqa_model_loader(model_name=args.unified_qa_model_name_or_path)
    device = device_setting(args=args)
    unified_qa_model = unified_qa_model.to(device)
    # # ################################################################################################################
    pred_answer = unified_qa_evaluation(unified_qa_model, unified_qa_tokenizer, raw_data, processed_data, device)
    with open(os.path.join(args.output_dir, args.model_name_or_path + '.json'), 'w') as fp:
        json.dump(pred_answer, fp)
    # #################################################################################################################
