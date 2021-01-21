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

# Example
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

def predict(raw_data, examples, features, pred_file, tokenizer, use_ent_ans=False):
    answer_dict = dict()
    sp_dict = dict()
    ids = list(examples.keys())
    max_sent_num = 0
    max_entity_num = 0
    q_type_counter = Counter()
    ans_type_counter = Counter()
    answer_no_match_cnt = 0

    for row in raw_data:
        qid = row['_id']
        feature = features[qid]
        example = examples[qid]
        q_type = feature.ans_type
        answer = row['answer']
        q_type = feature.ans_type
        ctx_names = [_[0] for _ in row['context']]
        # print(qid, answer, q_type)
        q_type_counter[q_type] += 1
        ans_type_counter[answer] += 1
        e_doc_tokens = example.doc_tokens
        q_tokens = example.question_tokens

        f_doc_input_ids = feature.doc_input_ids
        f_doc_tokens = feature.doc_tokens
        print(len(q_tokens), len(e_doc_tokens), len(f_doc_tokens), len(f_doc_input_ids))
        if len(e_doc_tokens) > len(f_doc_tokens):
            print(len(q_tokens), len(e_doc_tokens), len(f_doc_tokens), len(f_doc_input_ids))
            # print(e_doc_tokens)
            # print('-'*75)
            # print(f_doc_tokens)
            print('+' * 75)

    # max_sent_num = 0
    # max_entity_num = 0
    # q_type_counter = Counter()
    #
    # answer_no_match_cnt = 0
    # for i, qid in enumerate(ids):
    #     feature = features[qid]
    #     example = examples[qid]
    #     q_type = feature.ans_type
    #     print(qid, example)

    #     max_sent_num = max(max_sent_num, len(feature.sent_spans))
    #     max_entity_num = max(max_entity_num, len(feature.entity_spans))
    #     q_type_counter[q_type] += 1
    #
    #     def get_ans_from_pos(y1, y2):
    #         tok_to_orig_map = feature.token_to_orig_map
    #
    #         final_text = " "
    #         if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
    #             orig_tok_start = tok_to_orig_map[y1]
    #             orig_tok_end = tok_to_orig_map[y2]
    #
    #             ques_tok_len = len(example.question_tokens)
    #             if orig_tok_start < ques_tok_len and orig_tok_end < ques_tok_len:
    #                 ques_start_idx = example.question_word_to_char_idx[orig_tok_start]
    #                 ques_end_idx = example.question_word_to_char_idx[orig_tok_end] + len(example.question_tokens[orig_tok_end])
    #                 final_text = example.question_text[ques_start_idx:ques_end_idx]
    #             else:
    #                 orig_tok_start -= len(example.question_tokens)
    #                 orig_tok_end -= len(example.question_tokens)
    #                 ctx_start_idx = example.ctx_word_to_char_idx[orig_tok_start]
    #                 ctx_end_idx = example.ctx_word_to_char_idx[orig_tok_end] + len(example.doc_tokens[orig_tok_end])
    #                 final_text = example.ctx_text[example.ctx_word_to_char_idx[orig_tok_start]:example.ctx_word_to_char_idx[orig_tok_end]+len(example.doc_tokens[orig_tok_end])]
    #
    #         return final_text
    #         #return tokenizer.convert_tokens_to_string(tok_tokens)
    #
    #     answer_text = ''
    #     if q_type == 0 or q_type == 3:
    #         if len(feature.start_position) == 0 or len(feature.end_position) == 0:
    #             answer_text = ""
    #         else:
    #             #st, ed = example.start_position[0], example.end_position[0]
    #             #answer_text = example.ctx_text[example.ctx_word_to_char_idx[st]:example.ctx_word_to_char_idx[ed]+len(example.doc_tokens[example.end_position[0]])]
    #             answer_text = get_ans_from_pos(feature.start_position[0], feature.end_position[0])
    #             if normalize_answer(answer_text) != normalize_answer(example.orig_answer_text):
    #                 print("{} | {} | {} | {} | {}".format(qid, answer_text, example.orig_answer_text, feature.start_position[0], feature.end_position[0]))
    #                 answer_no_match_cnt += 1
    #         if q_type == 3 and use_ent_ans:
    #             ans_id = feature.answer_in_entity_ids[0]
    #             st, ed = feature.entity_spans[ans_id]
    #             answer_text = get_ans_from_pos(st, ed)
    #     elif q_type == 1:
    #         answer_text = 'yes'
    #     elif q_type == 2:
    #         answer_text = 'no'
    #
    #     answer_dict[qid] = answer_text
    #     cur_sp = []
    #     for sent_id in feature.sup_fact_ids:
    #         cur_sp.append(example.sent_names[sent_id])
    #     sp_dict[qid] = cur_sp
    #
    # # final_pred = {'answer': answer_dict, 'sp': sp_dict}
    # # json.dump(final_pred, open(pred_file, 'w'))

    # print("Maximum sentence num: {}".format(max_sent_num))
    # print("Maximum entity num: {}".format(max_entity_num))
    print("Question type: {}".format(q_type_counter))
    # print("Ans type: {}".format(ans_type_counter))
    # print("Answer doesnot match: {}".format(answer_no_match_cnt))

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
