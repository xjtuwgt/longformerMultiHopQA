# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
from pandas import DataFrame
from time import time
from longformerDataUtils.hotpotQAUtils import *
import swifter
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