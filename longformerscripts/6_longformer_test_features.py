import gzip
import pickle
import json
import torch
import numpy as np
import argparse
import os

from os.path import join
from collections import Counter

from model_envs import MODEL_CLASSES
from jd_mhqa.jd_data_processing import Example, InputFeatures, get_cached_filename
from csr_mhqa.utils import get_final_text
from envs import DATASET_FOLDER, OUTPUT_FOLDER
from eval.hotpot_evaluate_v1 import eval as hotpot_eval
from eval.hotpot_evaluate_v1 import normalize_answer

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

def predict(raw_data, examples, features, pred_file, tokenizer, use_ent_ans=False):
    answer_dict = dict()
    sp_dict = dict()
    ids = list(examples.keys())
    max_sent_num = 0
    max_entity_num = 0
    q_type_counter = Counter()
    ans_type_counter = Counter()
    answer_no_match_cnt = 0
    max_token_num = 0
    max_deep_token_num = 0
    token_num_list = []
    deep_token_num_list = []
    sent_num_list = []
    ent_num_list = []
    for row in raw_data:
        qid = row['_id']
        feature = features[qid]
        example = examples[qid]
        q_type = feature.ans_type
        answer = row['answer']
        q_type = feature.ans_type
        ctx_names = [_[0] for _ in row['context']]
        # print(qid, answer, q_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True, help='define output directory')
    parser.add_argument("--output_dir", type=str, required=True, help='define output directory')
    parser.add_argument("--graph_id", type=str, default="1", help='define output directory')

    # Other parameters
    parser.add_argument("--model_type", default="bert", type=str)
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
    print('Loading row data from: {}'.format(args.raw_data))
    print("Loading examples from: {}".format(cached_examples_file))
    print("Loading features from: {}".format(cached_features_file))
    print("Loading graphs from: {}".format(cached_graphs_file))

    pred_file = join(args.output_dir, 'pred.json')
    predict(raw_data, example_dict, feature_dict, pred_file, tokenizer, use_ent_ans=False)
    # metrics = hotpot_eval(pred_file, args.raw_data)
    # for key, val in metrics.items():
    #     print("{} = {}".format(key, val))
