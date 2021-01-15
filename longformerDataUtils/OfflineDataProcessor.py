import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
hotpot_path = '../data/hotpotqa/'
abs_hotpot_path = os.path.abspath(hotpot_path)
print('Abs hotpot Path = {}'.format(abs_hotpot_path))
from longformerDataUtils.ioutils import create_dir_if_not_exist
create_dir_if_not_exist(save_path=abs_hotpot_path, sub_folder='distractor_qa')
distractor_wiki_path = '../data/hotpotqa/distractor_qa'
abs_distractor_wiki_path = os.path.abspath(distractor_wiki_path)
print('Abs pre-process path = {}'.format(abs_distractor_wiki_path))

import swifter
from longformerDataUtils.hotpotQAUtils import *
from pandas import DataFrame
from time import time
from transformers import LongformerTokenizer
from longformerscripts.longformerUtils import PRE_TAINED_LONFORMER_BASE, get_hotpotqa_longformer_tokenizer
from longformerDataUtils.ioutils import HOTPOT_DevData_Distractor, HOTPOT_TrainData
import numpy as np
import pandas as pd

def Hotpot_Train_Dev_Data_Preprocess(data: DataFrame, tokenizer: LongformerTokenizer):
    # ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
    """
    Supporting_facts: pair of (title, sentence index) --> (str, int)
    Level: {easy, medium, hard}
    Question: query --> str
    Context: list of pair (title, text) --> (str, list(str))
    Answer: str
    Type: {comparison, bridge}
    """
    def train_dev_normalize_row(row):
        question, supporting_facts, context, answer = row['question'], row['supporting_facts'], row['context'], row['answer']
        norm_question = normalize_question(question=question)
        norm_answer = normalize_answer(ans=answer)
        ################################################################################################################
        doc_title2sent_len = dict([(title, len(sentences)) for title, sentences in context])
        filtered_supporting_facts = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                     if supp_sent_idx < doc_title2sent_len[supp_title]]
        ################################################################################################################
        positive_titles = set([x[0] for x in filtered_supporting_facts])
        assert len(positive_titles) == 2
        norm_context = []
        pos_idxs, neg_idxs = [], []
        for ctx_idx, ctx in enumerate(context):
            ctx_title, ctx_sentences = ctx
            norm_ctx_sentences = [normalize_text(sent) for sent in ctx_sentences]
            sent_labels = [0] * len(norm_ctx_sentences)
            if ctx_title in positive_titles:
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                pos_idxs.append(ctx_idx)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                count = 1
                supp_sent_flags = []
                for supp_title, supp_sent_idx in filtered_supporting_facts:
                    if ctx_title == supp_title:
                        supp_sent = norm_ctx_sentences[supp_sent_idx]
                        has_answer = answer_finder(norm_answer=norm_answer, supp_sent=supp_sent, tokenizer=tokenizer)
                        if has_answer:
                            count = count + 1
                            supp_sent_flags.append((supp_sent_idx, True))
                            sent_labels[supp_sent_idx] = 2
                        else:
                            supp_sent_flags.append((supp_sent_idx, False))
                            sent_labels[supp_sent_idx] = 1
                norm_context.append([ctx_title.lower(), norm_ctx_sentences, count, supp_sent_flags, True, sent_labels])
            else:
                neg_idxs.append(ctx_idx)
                norm_context.append([ctx_title.lower(), norm_ctx_sentences, 0, [], False, sent_labels])
        ################################################################################################################
        pos_counts = [norm_context[_][2] for _ in pos_idxs]
        assert len(pos_counts) == 2
        span_flag = norm_answer.strip() not in ['yes', 'no', 'noanswer']
        not_found_flag = False
        if span_flag:
            if sum(pos_counts) == 2: ### no answer founded in the supporting sentences
                not_found_flag = True
        if not_found_flag:
            norm_answer = 'noanswer'
            # print('id = {}'.format(row['_id']))
            span_flag = False
        ################################################################################################################
        return norm_question, filtered_supporting_facts, norm_context, norm_answer, pos_idxs, neg_idxs, span_flag, not_found_flag
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    normalized_names = ['norm_question', 'norm_supporting_facts', 'norm_ctx', 'norm_answer', 'p_ctx_idx', 'n_ctx_idx', 'span_flag', 'no_found']
    data[normalized_names] = data.swifter.apply(lambda row: pd.Series(train_dev_normalize_row(row)), axis=1)
    not_found_num = data[data['no_found']].shape[0]
    print('Step 1: Normalizing data takes {:.4f} seconds, no found records = {}'.format(time() - start_time, not_found_num))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, norm_ctxs, norm_answer, p_ctx_idxs = row['norm_question'], row['norm_ctx'], row['norm_answer'], row['p_ctx_idx']
        query_encode_ids, query_len = query_encoder(query=norm_question, tokenizer=tokenizer)
        ################################################################################################################
        ctx_encodes = []
        for ctx_idx, content in enumerate(norm_ctxs):
            if ctx_idx in p_ctx_idxs:
                encode_tuple = pos_context_encoder(norm_answer=norm_answer, content=content, tokenizer=tokenizer)
            else:
                encode_tuple = neg_context_encoder(content=content, tokenizer=tokenizer)
            ctx_encodes.append(encode_tuple)
        return query_encode_ids, ctx_encodes
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    encode_names = ['ques_encode', 'ctx_encode_list']
    data[encode_names] = data.swifter.apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Step 2: Encoding takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def long_combination_encoder(row):
        query_encode, ctx_encode, norm_answer = row['ques_encode'], row['ctx_encode_list'], row['norm_answer']
        if norm_answer.strip() in ['yes', 'no', 'noanswer']:  ## yes: 1, no/noanswer: 2, span = 0
            answer_type_label = np.array([1]) if norm_answer.strip() == 'yes' else np.array([2])
        else:
            answer_type_label = np.array([0])
        span_flag = row['span_flag']
        doc_infor, sent_infor, seq_infor, answer_infor = context_merge_longer(query_encode_ids=query_encode,
                                                                              context_tuple_list=ctx_encode,
                                                                              span_flag=span_flag)
        doc_labels, doc_ans_labels, doc_num, doc_len_array, doc_start_position, doc_end_position, doc_head_idx, doc_tail_idx = doc_infor
        sent_labels, sent_ans_labels, sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums = sent_infor
        concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask = seq_infor
        answer_pos_start, answer_pos_end, answer_pos_tuple = answer_infor
        return doc_labels, doc_ans_labels, doc_num, doc_len_array, doc_start_position, doc_end_position, doc_head_idx, doc_tail_idx, \
               sent_labels, sent_ans_labels, sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums, \
               concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask, \
               answer_pos_start, answer_pos_end, answer_pos_tuple, answer_type_label
    start_time = time()
    comb_res_col_names = ['doc_label', 'doc_ans_label', 'doc_num', 'doc_len', 'doc_start', 'doc_end', 'head_idx', 'tail_idx',
                          'sent_label', 'sent_ans_label', 'sent_num', 'sent_len', 'sent_start', 'sent_end', 'sent2doc', 'sentIndoc', 'doc_sent_num',
                          'ctx_encode', 'ctx_len', 'global_attn', 'token2sent', 'ans_mask', 'ans_start', 'ans_end', 'ans_pos_tups', 'answer_type']
    data[comb_res_col_names] = \
        data.swifter.apply(lambda row: pd.Series(long_combination_encoder(row)), axis=1)
    print('Step3: Combination takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    data_loader_res_columns = comb_res_col_names + ['_id', 'level']
    combined_data = data[data_loader_res_columns]
    norm_encode_col_names = normalized_names + encode_names + ['_id', 'level']
    ind_encoded_data = data[norm_encode_col_names]
    return data, combined_data, ind_encoded_data

def Hotpot_Test_Data_PreProcess(data: DataFrame, tokenizer: LongformerTokenizer):
    def test_normalize_row(row):
        question, context = row['question'], row['context']
        norm_question = normalize_question(question=question)
        ################################################################################################################
        norm_context = []
        for ctx_idx, ctx in enumerate(context):
            ctx_title, ctx_sentences = ctx
            norm_ctx_sentences = [normalize_text(sent) for sent in ctx_sentences]
            norm_context.append([ctx_title.lower(), norm_ctx_sentences])
        ################################################################################################################
        return norm_question, norm_context
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    normalized_names = ['norm_question', 'norm_ctx']
    data[normalized_names] = data.swifter.apply(lambda row: pd.Series(test_normalize_row(row)), axis=1)
    print('Step 1: Normalizing data takes {:.4f} seconds'.format(time() - start_time))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, norm_ctxs = row['norm_question'], row['norm_ctx']
        query_encode_ids, query_len = query_encoder(query=norm_question, tokenizer=tokenizer)
        ################################################################################################################
        ctx_encodes = []
        for ctx_idx, content in enumerate(norm_ctxs):
            encode_tuple = context_encoder(content=content, tokenizer=tokenizer)
            ctx_encodes.append(encode_tuple)
        return query_encode_ids, ctx_encodes
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    encode_names = ['ques_encode', 'ctx_encode_list']
    data[encode_names] = data.swifter.apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Step 2: Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def long_combination_encoder(row):
        query_encode, ctx_encode = row['ques_encode'], row['ctx_encode_list']
        doc_infor, sent_infor, seq_infor = test_context_merge_longer(query_encode_ids=query_encode,
                                                                              context_tuple_list=ctx_encode)
        doc_num, doc_len_array, doc_start_position, doc_end_position = doc_infor
        sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums = sent_infor
        concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask = seq_infor
        return doc_num, doc_len_array, doc_start_position, doc_end_position, \
               sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums, \
               concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask
    start_time = time()
    comb_res_col_names = ['doc_num', 'doc_len', 'doc_start', 'doc_end',
                          'sent_num', 'sent_len', 'sent_start', 'sent_end', 'sent2doc', 'sentIndoc', 'doc_sent_num',
                          'ctx_encode', 'ctx_len', 'global_attn', 'token2sent', 'ans_mask']
    data[comb_res_col_names] = \
        data.swifter.apply(lambda row: pd.Series(long_combination_encoder(row)), axis=1)
    print('Step 3: Combination takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    data_loader_res_columns = comb_res_col_names + ['_id']
    combined_data = data[data_loader_res_columns]
    norm_encode_col_names = normalized_names + encode_names + ['_id']
    ind_encoded_data = data[norm_encode_col_names]
    return data, combined_data, ind_encoded_data

########################################################################################################################
def hotpotqa_preprocess_example():
    start_time = time()
    longformer_tokenizer = get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE)
    dev_data, _ = HOTPOT_DevData_Distractor(path=abs_hotpot_path)
    print('*' * 100)
    all_dev_test_data, combined_dev_test_data_res, ind_dev_test_data_res = Hotpot_Test_Data_PreProcess(data=dev_data,
                                                                                                       tokenizer=longformer_tokenizer)
    print('Get {} dev-test records, encode records {}, tokenized records {}'.format(all_dev_test_data.shape[0],
                                                                                    combined_dev_test_data_res.shape[0],
                                                                                    ind_dev_test_data_res.shape[0]))
    all_dev_test_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_test_distractor_wiki_all.json'))
    combined_dev_test_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_test_distractor_wiki_combined.json'))
    ind_dev_test_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_test_distractor_wiki_ind.json'))
    print('*' * 100)
    dev_data, _ = HOTPOT_DevData_Distractor(path=abs_hotpot_path)
    all_dev_data, combined_dev_data_res, ind_dev_data_res = Hotpot_Train_Dev_Data_Preprocess(data=dev_data, tokenizer=longformer_tokenizer)
    print('Get {} dev records, encode records {} tokenized records {}'.format(all_dev_data.shape[0],
                                                                              combined_dev_data_res.shape[0],
                                                                              ind_dev_data_res.shape[0]))
    all_dev_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_dev_distractor_wiki_all.json'))
    combined_dev_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_dev_distractor_wiki_combined.json'))
    ind_dev_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_dev_distractor_wiki_ind.json'))
    print('*' * 100)
    train_data, _ = HOTPOT_TrainData(path=abs_hotpot_path)
    all_train_data, combined_train_data_res, ind_train_data_res = Hotpot_Train_Dev_Data_Preprocess(data=train_data, tokenizer=longformer_tokenizer)
    print('Get {} training records, encode records {} tokenized records {}'.format(all_train_data.shape[0],
                                                                                   combined_train_data_res.shape[0],
                                                                                   ind_train_data_res.shape[0]))
    all_train_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_train_distractor_wiki_all.json'))
    combined_train_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_train_distractor_wiki_combined.json'))
    ind_train_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_train_distractor_wiki_ind.json'))
    print('Runtime = {:.4f} seconds'.format(time() - start_time))
    print('*' * 100)

def Gold_Hotpot_Train_Dev_Data_Collection(data: DataFrame):
    def pos_context_extraction(row):
        supporting_facts, contexts = row['supporting_facts'], row['context']
        positive_titles = set([x[0] for x in supporting_facts])
        pos_context = []
        for ctx_idx, ctx in enumerate(contexts):  ## Original ctx index, record the original index order
            title, text = ctx[0], ctx[1]
            if title in positive_titles:
                pos_context.append([title, text])
        assert len(pos_context) == 2
        return pos_context, len(pos_context)
    data[['context', 'pos_num']] = data.swifter.apply(lambda row: pd.Series(pos_context_extraction(row)), axis=1)
    return data

def gold_doc_hotpotqa_extraction_example():
    print('Pre-processing gold data...')
    longformer_tokenizer = get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE)
    start_time = time()
    dev_data, _ = HOTPOT_DevData_Distractor(path=abs_hotpot_path)
    print('*' * 75)
    gold_dev_data = Gold_Hotpot_Train_Dev_Data_Collection(data=dev_data)
    print('Get {} gold dev-test records'.format(gold_dev_data.shape[0]))
    gold_dev_data.to_json(os.path.join(hotpot_path, 'gold_hotpot_dev_distractor_v1.json'))
    print('Runtime = {:.4f} seconds to get gold documents'.format(time() - start_time))
    all_gold_test_data, combined_gold_test_data_res, ind_gold_norm_test_data = Hotpot_Test_Data_PreProcess(data=gold_dev_data,
                                                                                                        tokenizer=longformer_tokenizer)
    print('Get {} dev-test records, encode records {}, normalized records {}'.format(all_gold_test_data.shape[0],
                                                                                     combined_gold_test_data_res.shape[0],
                                                                                     ind_gold_norm_test_data.shape[0]))
    all_gold_test_data.to_json(os.path.join(abs_distractor_wiki_path, 'gold_hotpot_test_distractor_wiki_all.json'))
    combined_gold_test_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'gold_hotpot_test_distractor_wiki_combined.json'))
    ind_gold_norm_test_data.to_json(os.path.join(abs_distractor_wiki_path, 'gold_hotpot_test_distractor_wiki_ind.json'))
    print('*' * 75)
    all_gold_dev_data, combined_gold_dev_data_res, ind_gold_norm_dev_data = Hotpot_Train_Dev_Data_Preprocess(data=gold_dev_data,
                                                                                                        tokenizer=longformer_tokenizer)
    print('Get {} dev-test records, encode records {}, normalized records {}'.format(all_gold_dev_data.shape[0],
                                                                                     combined_gold_dev_data_res.shape[0],
                                                                                     ind_gold_norm_dev_data.shape[0]))
    all_gold_dev_data.to_json(os.path.join(abs_distractor_wiki_path, 'gold_hotpot_dev_distractor_wiki_all.json'))
    combined_gold_dev_data_res.to_json(os.path.join(abs_distractor_wiki_path, 'gold_hotpot_dev_distractor_wiki_combined.json'))
    ind_gold_norm_dev_data.to_json(os.path.join(abs_distractor_wiki_path, 'gold_hotpot_dev_distractor_wiki_ind.json'))
    print('*' * 75)

if __name__ == '__main__':
    hotpotqa_preprocess_example()
    gold_doc_hotpotqa_extraction_example()
    print()