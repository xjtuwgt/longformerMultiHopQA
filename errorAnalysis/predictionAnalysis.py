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

from model_envs import MODEL_CLASSES
from jd_mhqa.jd_data_processing import Example, InputFeatures, get_cached_filename

def exmple_infor_collection(example: Example):
    # Example
    # self.qas_id = qas_id
    #         self.qas_type = qas_type
    #         self.question_tokens = question_tokens
    #         self.doc_tokens = doc_tokens
    #         self.question_text = question_text
    #         self.sent_num = sent_num
    #         self.sent_names = sent_names
    #         self.para_names = para_names
    #         self.sup_fact_id = sup_fact_id
    #         self.sup_para_id = sup_para_id
    #         self.ques_entities_text = ques_entities_text
    #         self.ctx_entities_text = ctx_entities_text
    #         self.para_start_end_position = para_start_end_position
    #         self.sent_start_end_position = sent_start_end_position
    #         self.ques_entity_start_end_position = ques_entity_start_end_position
    #         self.ctx_entity_start_end_position = ctx_entity_start_end_position
    #         self.question_word_to_char_idx = question_word_to_char_idx
    #         self.ctx_text = ctx_text
    #         self.ctx_word_to_char_idx = ctx_word_to_char_idx
    #         self.edges = edges
    #         self.orig_answer_text = orig_answer_text
    #         self.answer_in_ques_entity_ids = answer_in_ques_entity_ids
    #         self.answer_in_ctx_entity_ids = answer_in_ctx_entity_ids
    #         self.answer_candidates_in_ctx_entity_ids= answer_candidates_in_ctx_entity_ids
    #         self.start_position = start_position
    #         self.end_position = end_position
    doc_tokens = example.doc_tokens
    query_tokens = example.question_tokens
    sent_num = example.sent_num
    sent_start_end_position = example.sent_start_end_position
    ent_start_end_position = example.ctx_entity_start_end_position
    print(sent_num, len(sent_start_end_position))

    return

def feature_infor_collection(feature: InputFeatures):
    # features
    # self.qas_id = qas_id
    #         self.doc_tokens = doc_tokens
    #         self.doc_input_ids = doc_input_ids
    #         self.doc_input_mask = doc_input_mask
    #         self.doc_segment_ids = doc_segment_ids
    #
    #         self.query_tokens = query_tokens
    #         self.query_input_ids = query_input_ids
    #         self.query_input_mask = query_input_mask
    #         self.query_segment_ids = query_segment_ids
    #
    #         self.para_spans = para_spans
    #         self.sent_spans = sent_spans
    #         self.entity_spans = entity_spans
    #         self.q_entity_cnt = q_entity_cnt
    #         self.sup_fact_ids = sup_fact_ids
    #         self.sup_para_ids = sup_para_ids
    #         self.ans_type = ans_type
    #
    #         self.edges = edges
    #         self.token_to_orig_map = token_to_orig_map
    #         self.orig_answer_text = orig_answer_text
    #         self.answer_in_entity_ids = answer_in_entity_ids
    #         self.answer_candidates_ids = answer_candidates_ids
    #
    #         self.start_position = start_position
    #         self.end_position = end_position
    doc_tokens = feature.doc_tokens
    sent_spans = feature.sent_spans
    ent_spans = feature.entity_spans
    return

def set_comparison(prediction_list, true_list):
    def em():
        if len(prediction_list) != len(true_list):
            return False
        for pred in prediction_list:
            if pred not in true_list:
                return False
        return True
    if em():
        return 'em'

    is_subset = set(true_list).issubset(set(prediction_list))
    if is_subset:
        return 'super_of_gold'
    is_super_set = set(prediction_list).issubset(set(true_list))
    if is_super_set:
        return 'sub_of_gold'
    return 'others'



def error_analysis(raw_data, examples, features, predictions, tokenizer, use_ent_ans=False):
    yes_no_span_predictions = []
    yes_no_span_true = []
    prediction_ans_type_counter = Counter()
    prediction_sent_type_counter = Counter()
    prediction_para_type_counter = Counter()
    pred_ans_type_list = []
    pred_sent_type_list = []
    pred_doc_type_list = []
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for row in raw_data:
        qid = row['_id']
        sp_predictions = predictions['sp'][qid]
        sp_predictions = [(x[0], x[1]) for x in sp_predictions]
        ans_prediction = predictions['answer'][qid]

        raw_answer = row['answer']
        raw_answer = normalize_answer(raw_answer)
        ans_prediction = normalize_answer(ans_prediction)
        sp_golds = row['supporting_facts']
        sp_golds = [(x[0], x[1]) for x in sp_golds]
        sp_sent_type = set_comparison(prediction_list=sp_predictions, true_list=sp_golds)
        ###+++++++++
        prediction_sent_type_counter[sp_sent_type] +=1
        pred_sent_type_list.append(sp_sent_type)
        ###+++++++++
        sp_para_golds = list(set([_[0] for _ in sp_golds]))
        sp_para_preds = list(set([_[0] for _ in sp_predictions]))
        para_type = set_comparison(prediction_list=sp_para_preds, true_list=sp_para_golds)
        prediction_para_type_counter[para_type] += 1
        pred_doc_type_list.append(para_type)
        ###+++++++++

        if raw_answer not in ['yes', 'no']:
            yes_no_span_true.append('span')
        else:
            yes_no_span_true.append(raw_answer)

        if ans_prediction not in ['yes', 'no']:
            yes_no_span_predictions.append('span')
        else:
            yes_no_span_predictions.append(ans_prediction)

        ans_type = 'em'
        if raw_answer not in ['yes', 'no']:
            if raw_answer == ans_prediction:
                ans_type = 'em'
            elif raw_answer in ans_prediction:
                # print('{}: {} |{}'.format(qid, raw_answer, ans_prediction))
                # print('-'*75)
                ans_type = 'super_of_gold'
            elif ans_prediction in raw_answer:
                # print('{}: {} |{}'.format(qid, raw_answer, ans_prediction))
                # print('-'*75)
                ans_type = 'sub_of_gold'
            else:
                # inter_res_len = len(set(ans_prediction).intersection(raw_answer))
                # # print(inter_res_len)
                # if inter_res_len > max(len(ans_prediction), len(raw_answer)) * 0.5:
                #     prediction_ans_type_counter['inter0.5'] += 1
                #     # print('{}: {} |{}'.format(qid, raw_answer, ans_prediction))
                #     # print('-'*75)
                # else:
                #     prediction_ans_type_counter['others'] += 1
                #     print('{}: {} |{}'.format(qid, raw_answer, ans_prediction))
                #     print('-'*75)
                ans_type = 'others'
                # print('{}: {} |{}'.format(qid, raw_answer, ans_prediction))
                # print('-' * 75)
        else:
            if raw_answer == ans_prediction:
                ans_type = 'em'
            else:
                ans_type = 'others'

        prediction_ans_type_counter[ans_type] += 1
        pred_ans_type_list.append(ans_type)


    print(len(pred_sent_type_list), len(pred_ans_type_list), len(pred_doc_type_list))

    result_types = ['em', 'sub_of_gold', 'super_of_gold', 'others']
    conf_matrix = confusion_matrix(yes_no_span_true, yes_no_span_predictions, labels=["yes", "no", "span"])
    conf_ans_sent_matrix = confusion_matrix(pred_ans_type_list, pred_sent_type_list, labels=result_types)
    print('Ans type conf matrix:\n{}'.format(conf_matrix))
    print('Type conf matrix:\n{}'.format(conf_ans_sent_matrix))

    print("Ans prediction type: {}".format(prediction_ans_type_counter))
    print("Sent prediction type: {}".format(prediction_sent_type_counter))
    print("Para prediction type: {}".format(prediction_para_type_counter))

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
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")

    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")

    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    cached_examples_file = os.path.join(args.input_dir,
                                        get_cached_filename('examples', args))
    cached_features_file = os.path.join(args.input_dir,
                                        get_cached_filename('features',  args))
    cached_graphs_file = os.path.join(args.input_dir,
                                     get_cached_filename('graphs', args))

    examples = pickle.load(gzip.open(cached_examples_file, 'rb'))
    features = pickle.load(gzip.open(cached_features_file, 'rb'))
    graph_dict = pickle.load(gzip.open(cached_graphs_file, 'rb'))

    example_dict = { example.qas_id: example for example in examples}
    feature_dict = { feature.qas_id: feature for feature in features}

    with open(args.raw_data, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)

    pred_results_file = os.path.join(args.pred_dir, args.model_type, 'pred.json')
    with open(pred_results_file, 'r', encoding='utf-8') as reader:
        pred_data = json.load(reader)

    print('Loading predictions from: {}'.format(pred_results_file))
    print('Loading raw data from: {}'.format(args.raw_data))
    print("Loading examples from: {}".format(cached_examples_file))
    print("Loading features from: {}".format(cached_features_file))
    print("Loading graphs from: {}".format(cached_graphs_file))

    error_analysis(raw_data, example_dict, feature_dict, pred_data, tokenizer, use_ent_ans=False)
    # metrics = hotpot_eval(pred_file, args.raw_data)
    # for key, val in metrics.items():
    #     print("{} = {}".format(key, val))
