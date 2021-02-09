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
    instance_variables = vars(feature)
    for key, value in instance_variables.items():
        print(key)

    # print(instance_variables)
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
    is_empty_set = len(set(prediction_list).intersection(set(true_list)))==0
    if is_empty_set:
        return 'no_over_lap'
    return 'others'

def data_analysis(raw_data, examples, features, tokenizer, use_ent_ans=False):
    # example_sent_num_list = []
    # example_ent_num_list = []
    # example_ctx_num_list = []
    example_doc_recall_list = []
    feature_doc_recall_list = []

    example_sent_recall_list = []
    feature_sent_recall_list = []
    trim_yes_no_count = 0
    for row in raw_data:
        qid = row['_id']
        answer = row['answer']
        gold_doc_names = list(set([_[0] for _ in row['supporting_facts']]))
        raw_context = row['context']
        raw_supp_sents = [(x[0], x[1]) for x in row['supporting_facts']]
        ################################################################################################################
        feature = features[qid]
        feature_dict = vars(feature)
        doc_input_ids = feature_dict['doc_input_ids']
        assert len(doc_input_ids) == 512
        # doc_512_context = tokenizer.decode(doc_input_ids, skip_special_tokens=True)
        para_spans = feature_dict['para_spans']
        trim_doc_names = [_[2] for _ in para_spans]
        feature_em_recall = recall_computation(prediction=trim_doc_names, gold=gold_doc_names)
        feature_doc_recall_list.append(feature_em_recall)
        trim_sent_spans = feature_dict['sent_spans']

        ################################################################################################################
        # for key, value in feature_dict.items():
        #     print('F: {}\t{}'.format(key, value))
        ################################################################################################################
        example = examples[qid]
        example_dict = vars(example)
        example_doc_names = example_dict['para_names']
        em_recall = recall_computation(prediction=example_doc_names, gold=gold_doc_names)
        example_doc_recall_list.append(em_recall)
        example_sent_names = example_dict['sent_names']
        em_sent_recall = recall_computation(prediction=example_sent_names, gold=raw_supp_sents)
        example_sent_recall_list.append(em_sent_recall)
        trim_span_sent_names = [example_sent_names[i] for i in range(len(trim_sent_spans))]
        trim_em_sent_recall = recall_computation(prediction=trim_span_sent_names, gold=raw_supp_sents)
        feature_sent_recall_list.append(trim_em_sent_recall)
        if trim_em_sent_recall != 1:
            if answer in ['yes', 'no']:
                trim_yes_no_count += 1
        ################################################################################################################
        # for key, value in example_dict.items():
        #     print('E:{}\t{}'.format(key, value))
        ################################################################################################################
        # print(len(example_doc_names), len(para_spans))
        # if len(example_doc_names) > len(para_spans):
        #     print(qid)
        #     print('Example context:\n{}'.format(example_dict['ctx_text']))
        #     print('-' * 100)
        #     print('Feature context:\n{}'.format(tokenizer.decode(doc_input_ids, skip_special_tokens=True)))
        #     print('+' * 100)
        #     cut_para_names = [x for x in example_doc_names if x not in trim_doc_names]
        #     print(len(example_doc_names), len(para_spans), len(cut_para_names))
        #     for c_idx, cut_para in enumerate(cut_para_names):
        #         for ctx_idx, ctx in enumerate(raw_context):
        #             if cut_para == ctx[0]:
        #                 print('Cut para {}:\n{}'.format(c_idx, ctx[1]))
        #     print('*'*100)

        # print('$' * 100)
        # if len(example_sent_names) > len(trim_sent_spans):
        #     print(qid)
        #     break

    print('Example doc recall: {}'.format(sum(example_doc_recall_list)/len(example_doc_recall_list)))
    print('Example doc recall (512 trim): {}'.format(sum(feature_doc_recall_list)/len(feature_doc_recall_list)))
    print('Example sent recall: {}'.format(sum(example_sent_recall_list) / len(example_sent_recall_list)))
    print('Example sent recall (512 trim): {}'.format(sum(feature_sent_recall_list) / len(feature_sent_recall_list)))
    print('Trim yes no : {}'.format(trim_yes_no_count))

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
                ans_pred_tokens = ans_prediction.split(' ')
                ans_raw_tokens = raw_answer.split(' ')
                is_empty_set = len(set(ans_pred_tokens).intersection(set(ans_raw_tokens))) == 0
                if is_empty_set:
                    ans_type = 'no_over_lap'
                else:
                    ans_type = 'others'
        else:
            if raw_answer == ans_prediction:
                ans_type = 'em'
            else:
                ans_type = 'others'

        prediction_ans_type_counter[ans_type] += 1
        pred_ans_type_list.append(ans_type)


    print(len(pred_sent_type_list), len(pred_ans_type_list), len(pred_doc_type_list))

    result_types = ['em', 'sub_of_gold', 'super_of_gold', 'no_over_lap', 'others']
    conf_matrix = confusion_matrix(yes_no_span_true, yes_no_span_predictions, labels=["yes", "no", "span"])
    conf_ans_sent_matrix = confusion_matrix(pred_sent_type_list, pred_ans_type_list, labels=result_types)
    print('*' * 75)
    print('Ans type conf matrix:\n{}'.format(conf_matrix))
    print('*' * 75)
    print('Type conf matrix:\n{}'.format(conf_ans_sent_matrix))
    print('*' * 75)
    print("Ans prediction type: {}".format(prediction_ans_type_counter))
    print("Sent prediction type: {}".format(prediction_sent_type_counter))
    print("Para prediction type: {}".format(prediction_para_type_counter))
    print('*' * 75)

    conf_matrix_para_vs_sent = confusion_matrix(pred_doc_type_list, pred_sent_type_list, labels=result_types)
    print('Para Type vs Sent Type conf matrix:\n{}'.format(conf_matrix_para_vs_sent))
    print('*' * 75)
    conf_matrix_para_vs_ans = confusion_matrix(pred_doc_type_list, pred_ans_type_list, labels=result_types)
    print('Para Type vs Sent Type conf matrix:\n{}'.format(conf_matrix_para_vs_ans))
    # pred_sent_para_type_counter = Counter()
    # for (sent_type, para_type) in zip(pred_doc_type_list, pred_sent_type_list):
    #     pred_sent_para_type_counter[(sent_type, para_type)] += 1
    # print('*' * 75)
    # for key, value in dict(pred_sent_para_type_counter).items():
    #     print('{} vs {}: {}'.format(key[0], key[1], value))
    # print('Para sent type: {}'.format(pred_sent_para_type_counter))

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
    parser.add_argument("--at_model_name_or_path", default=None, type=str, required=True,
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
    # pred_results_file = os.path.join(args.pred_dir, args.model_name_or_path, 'tmp.json')
    with open(pred_results_file, 'r', encoding='utf-8') as reader:
        pred_data = json.load(reader)

    print('Loading predictions from: {}'.format(pred_results_file))
    print('Loading raw data from: {}'.format(args.raw_data))
    print("Loading examples from: {}".format(cached_examples_file))
    print("Loading features from: {}".format(cached_features_file))
    print("Loading graphs from: {}".format(cached_graphs_file))

    error_analysis(raw_data, example_dict, feature_dict, pred_data, tokenizer, use_ent_ans=False)
    data_analysis(raw_data, example_dict, feature_dict, tokenizer, use_ent_ans=False)
    # metrics = hotpot_eval(pred_file, args.raw_data)
    # for key, val in metrics.items():
    #     print("{} = {}".format(key, val))
