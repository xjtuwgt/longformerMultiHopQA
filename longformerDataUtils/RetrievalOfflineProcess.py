import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
hotpot_path = '../data/hotpotqa/'
abs_hotpot_path = os.path.abspath(hotpot_path)
print('Abs-HotPotQA-Path = {}'.format(abs_hotpot_path))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from dataUtils.ioutils import HOTPOT_Retrieval_Data, HOTPOT_DevData_Distractor
from dataUtils.ioutils import create_dir_if_not_exist
import pandas as pd
from pandas import DataFrame
from time import time
from dataUtils.hotpotQAUtils import *
from dataUtils.ioutils import loadJSONData
from evaluationUtils.hotpotEvaluationUtils import recall_computation
from dataUtils.OfflineDataProcessor import get_hotpotqa_longformer_tokenizer, PRE_TAINED_LONFORMER_BASE
from dataUtils.OfflineDataProcessor import Hotpot_Test_Data_PreProcess
import swifter
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
create_dir_if_not_exist(save_path=abs_hotpot_path, sub_folder='distractor_qa')
topk_distractor_wiki_path = '../data/hotpotqa/distractor_qa'
abs_topk_distractor_wiki_path = os.path.abspath(topk_distractor_wiki_path)
print('Abs-Topk-pre-process Path = {}'.format(abs_topk_distractor_wiki_path))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Hotpot_Retrieval_Train_Dev_Data_Preprocess(data: DataFrame, tokenizer: LongformerTokenizer):
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
        filtered_supporting_facts = []
        for supp_title, supp_sent_idx in supporting_facts:
            if supp_title in doc_title2sent_len:
                if supp_sent_idx < doc_title2sent_len[supp_title]:
                    filtered_supporting_facts.append((supp_title, supp_sent_idx))
            else:
                filtered_supporting_facts.append((supp_title, supp_sent_idx))
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
        span_flag = norm_answer.strip() not in ['yes', 'no', 'noanswer']
        pos_counts = [norm_context[_][2] for _ in pos_idxs]
        not_found_flag = False
        if len(pos_counts) == 2:
            if span_flag:
                if sum(pos_counts) == 2:  ### no answer founded in the supporting sentences
                    not_found_flag = True
        elif len(pos_counts) == 1:
            if span_flag:
                if pos_counts[0] < 2:
                    not_found_flag = True
        else:
            not_found_flag = True
        if not_found_flag:
            norm_answer = 'noanswer'
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
        doc_infor, sent_infor, seq_infor, answer_infor = dev_context_merge_longer(query_encode_ids=query_encode,
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
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def CombineRetrievalWithDevData():
    ir_data = HOTPOT_Retrieval_Data()
    orig_data, _ = HOTPOT_DevData_Distractor()
    assert ir_data.shape[0] == orig_data.shape[0]
    ir_data['e_id'] = range(0, ir_data.shape[0])
    orig_data['e_id'] = range(0, orig_data.shape[0])
    merge_data = pd.concat([ir_data.set_index('e_id'), orig_data.set_index('e_id')], axis=1, join='inner')
    assert merge_data.shape[0] == ir_data.shape[0]
    merge_data.reset_index(drop=True, inplace=True)
    return merge_data
def IR_Metrics_Computation(data: DataFrame, topk=3, model_num=9):
    def metrics_computation(row):
        support_doc_titles = [x[0] for x in row['supporting_facts']]
        ctx_titles = [x[0] for x in row['context']]
        ctx_len = len(ctx_titles)
        model_topk_recall = []
        for model_idx in range(model_num):
            model_i = model_idx + 1
            preds_i = row['model_{}'.format(model_i)].tolist()
            filtered_preds_i = [x for x in preds_i if x < ctx_len]
            if len(ctx_titles) <= topk:
                topk_titles = ctx_titles
            else:
                topk_titles = [ctx_titles[filtered_preds_i[_]] for _ in range(topk)]
            recall_i = recall_computation(topk_titles, support_doc_titles)
            model_topk_recall.append(recall_i==1)
        return model_topk_recall
    recall_names = ['recall_{}_{}'.format(topk, _ + 1) for _ in range(model_num)]
    data[recall_names] = data.apply(lambda row: pd.Series(metrics_computation(row)), axis=1)
    recall_data = data[recall_names]
    topk_recall_mean = recall_data.mean()
    print(topk_recall_mean)
    max_idx = topk_recall_mean.argmax()
    max_recall = topk_recall_mean.max()
    max_column_name = recall_names[max_idx]
    max_model_name = 'model_' + max_column_name.split('_')[-1]
    print('Model {} with max recall: {} with topk = {}'.format(max_model_name, max_recall, topk))
    def retrieval_extraction(row):
        context = row['context']
        ctx_titles = [x[0] for x in context]
        ctx_len = len(ctx_titles)
        preds = row[max_model_name].tolist()
        filtered_preds_i = [x for x in preds if x < ctx_len]
        recall = row[max_column_name]
        if len(ctx_titles) <= topk:
            topk_preds = filtered_preds_i
        else:
            topk_preds = [filtered_preds_i[_] for _ in range(topk)]
        ######################################
        topk_preds = sorted(topk_preds)
        ######################################
        topk_context = [context[doc_idx] for doc_idx in topk_preds]
        topk_context = tuple(topk_context)
        return topk_context, tuple(topk_preds), recall
    topk_context = ['top{}_context'.format(topk), 'top{}_idx'.format(topk), 'top{}_R'.format(topk)]
    data[topk_context] = data.apply(lambda row: pd.Series(retrieval_extraction(row)), axis=1)
    res_column_names = ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type'] + topk_context
    res_data = data[res_column_names]
    rename_dict = {'context': 'orig_context', 'top{}_context'.format(topk): 'context'}
    res_data = res_data.rename(columns=rename_dict)
    return res_data, max_model_name, max_recall
def topk_dev_data_extraction(max_topk=6):
    start_time = time()
    merge_data = CombineRetrievalWithDevData()
    print('Merging data takes {:.4f} seconds'.format(time() - start_time))
    print('*'*75)
    topk_dev_data_names = []
    for topk in range(2, max_topk):
        start_time = time()
        topk_data, topk_model, topk_recall = IR_Metrics_Computation(data=merge_data, topk=topk)
        print('Runtime = {:.4f} seconds to get top {} documents'.format(time() - start_time, topk))
        top_dev_data_name_i = 'hotpot_dev_distractor_top{}_{}_{:.3f}'.format(topk, topk_model, topk_recall)
        topk_data.to_json(os.path.join(abs_hotpot_path, top_dev_data_name_i + '.json'))
        print('Get {} dev-test records'.format(topk_data.shape[0]))
        topk_dev_data_names.append(top_dev_data_name_i)
        print('*' * 75)
    return topk_dev_data_names
########################################################################################################################
def hotpotqa_preprocess_example(max_topk=6):
    ####################################################################################################################
    longformer_tokenizer = get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE)
    ####################################################################################################################
    topk_dev_data_names = topk_dev_data_extraction(max_topk=max_topk)
    ####################################################################################################################
    print('+' * 75)
    for topk_dev_name in topk_dev_data_names:
        start_time = time()
        topk_dev_data_i = loadJSONData(PATH=abs_hotpot_path, json_fileName=topk_dev_name + '.json')
        all_topk_dev_data, combined_topk_dev_data_res, ind_topk_norm_dev_data_res = Hotpot_Retrieval_Train_Dev_Data_Preprocess(data=topk_dev_data_i,
                                                                                     tokenizer=longformer_tokenizer)
        print('Get {} dev records, encode records {} tokenized records {}'.format(all_topk_dev_data.shape[0],
                                                                                  combined_topk_dev_data_res.shape[0],
                                                                                  ind_topk_norm_dev_data_res.shape[0]))
        all_topk_dev_data.to_json(os.path.join(abs_topk_distractor_wiki_path, topk_dev_name + '_all.json'))
        combined_topk_dev_data_res.to_json(os.path.join(abs_topk_distractor_wiki_path, topk_dev_name + '_combined.json'))
        ind_topk_norm_dev_data_res.to_json(
            os.path.join(abs_topk_distractor_wiki_path, topk_dev_name + '_ind.json'))
        print('=' * 100)
        all_topk_test_data, combined_topk_test_data_res, ind_topk_norm_test_data_res = Hotpot_Test_Data_PreProcess(data=topk_dev_data_i,
                                                                                     tokenizer=longformer_tokenizer)
        print('Get {} test records, encode records {} tokenized records {}'.format(all_topk_test_data.shape[0],
                                                                                  combined_topk_test_data_res.shape[0],
                                                                                  ind_topk_norm_test_data_res.shape[0]))
        all_topk_test_data.to_json(os.path.join(abs_topk_distractor_wiki_path, topk_dev_name + '_test_all.json'))
        combined_topk_test_data_res.to_json(os.path.join(abs_topk_distractor_wiki_path, topk_dev_name + '_test_combined.json'))
        ind_topk_norm_test_data_res.to_json(
            os.path.join(abs_topk_distractor_wiki_path, topk_dev_name + '_test_ind.json'))
        print('Runtime = {:.4f} seconds'.format(time() - start_time))

if __name__ == '__main__':
    hotpotqa_preprocess_example(max_topk=5)
    print()