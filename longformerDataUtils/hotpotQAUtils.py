import string
import itertools
import operator
import numpy as np
import torch
import random
import torch.nn.functional as F
from scipy.linalg import block_diag
from transformers import LongformerTokenizer
########################################################################################################################
SPECIAL_QUERY_START = '<s>' ### for query marker
SPECIAL_QUERY_END = '</s>' ### for query marker
SPECIAL_TITLE_START = '<s>' ### for document maker
SPECIAL_TITLE_END = '</s>' ## for setence marker
SPECIAL_SENTENCE_TOKEN = '</s>'
CLS_TOKEN = '<s>'
########################################################################################################################
MAX_DOC_NUM = 10
MAX_SENT_NUM = 150
MAX_TOKEN_NUM = 4096
########################################################################################################################
def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    question = question.lower()
    return question

def normalize_text(text: str) -> str:
    text = ' ' + text.lower().strip() ###adding the ' ' is important to make the consist encoder, for roberta tokenizer
    return text

def normalize_answer(ans: str) -> str:
    def remove_punc(text):
        return text.strip(string.punctuation)
    def lower(text):
        return text.lower()
    answer_punc_trim = remove_punc(lower(ans))
    ############################################################
    if answer_punc_trim.strip() in {'yes', 'no', 'no_answer'}:
        answer_punc_trim = ans
    ############################################################
    if len(answer_punc_trim) == 0:
        norm_answer = normalize_text(text=ans)  ## add a space between the string sequence
    else:
        norm_answer = normalize_text(text=answer_punc_trim)
    return norm_answer

def answer_finder(norm_answer: str, supp_sent: str, tokenizer: LongformerTokenizer):
    if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
        str_has_answer = answer_span_str_finder(norm_answer, supp_sent)  ### string based checker
        if str_has_answer:
            encode_has_answer, _, _ = answer_span_token_finder(norm_answer, supp_sent, tokenizer)  ### encoded ids matching
            if encode_has_answer:
                return True
            else:
                encode_has_answer, _, _ = answer_span_token_finder(norm_answer.strip(), supp_sent, tokenizer)
                if encode_has_answer:
                    return True
                else:
                    return False
        else:
            return False
    else:
        return False

def answer_span_str_finder(answer: str, sentence: str) -> int:
    find_idx = sentence.find(answer.strip())
    return find_idx >=0

def answer_span_token_finder(norm_answer: str, sentence: str, tokenizer: LongformerTokenizer):
    answer_encode_ids = tokenizer.encode(text=norm_answer, add_special_tokens=False)
    sentence_encode_ids = tokenizer.encode(text=sentence, add_special_tokens=False)
    idx = sub_list_finder(target=answer_encode_ids, source=sentence_encode_ids)
    flag = idx >= 0
    return flag, answer_encode_ids, sentence_encode_ids

def sub_list_finder(target: list, source: list) -> int:
    if len(target) > len(source):
        return -1
    t_len = len(target)
    def equal_list(a_list, b_list):
        assert len(a_list) == len(b_list)
        for j in range(len(a_list)):
            if a_list[j] != b_list[j]:
                return False
        return True
    for i in range(len(source) - len(target) + 1):
        temp = source[i:(i+t_len)]
        is_equal = equal_list(target, temp)
        if is_equal:
            return i
    return -1
########################################################################################################################
def query_encoder(query: str, tokenizer: LongformerTokenizer):
    query_res = CLS_TOKEN + SPECIAL_QUERY_START + query + SPECIAL_QUERY_END
    query_tokens = tokenizer.tokenize(text=query_res)
    query_encode_ids = tokenizer.encode(text=query_tokens, add_special_tokens=False)
    assert len(query_tokens) == len(query_encode_ids)
    query_len = len(query_encode_ids)
    return query_encode_ids, query_len

def document_encoder(title: str, doc_sents: list, tokenizer: LongformerTokenizer):
    title_res = SPECIAL_TITLE_START + title + SPECIAL_TITLE_END##
    title_tokens = tokenizer.tokenize(text=title_res)
    title_encode_ids = tokenizer.encode(text=title_tokens, add_special_tokens=False)
    assert len(title_tokens) == len(title_encode_ids)
    title_len = len(title_encode_ids)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encode_id_lens = []
    encode_id_lens.append(title_len)
    doc_encode_id_list = []
    doc_encode_id_list.append(title_encode_ids)
    for sent_idx, sent_text in enumerate(doc_sents):
        sent_text_res = sent_text + SPECIAL_SENTENCE_TOKEN
        sent_tokens = tokenizer.tokenize(text=sent_text_res)
        sent_encode_ids = tokenizer.encode(text=sent_tokens, add_special_tokens=False)
        assert len(sent_tokens) == len(sent_encode_ids)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_encode_id_list.append(sent_encode_ids)
        sent_len = len(sent_encode_ids)
        encode_id_lens.append(sent_len)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    doc_sent_len_cum_list = list(itertools.accumulate(encode_id_lens, operator.add))
    sent_start_end_pair = [(doc_sent_len_cum_list[i], doc_sent_len_cum_list[i + 1] - 1) for i in range(len(encode_id_lens) - 1)]
    doc_encode_ids = list(itertools.chain.from_iterable(doc_encode_id_list))
    assert len(doc_encode_ids) == doc_sent_len_cum_list[-1]
    assert len(sent_start_end_pair) == len(doc_sents)
    return doc_encode_ids, sent_start_end_pair, len(doc_encode_ids), title_len
########################################################################################################################
def pos_context_encoder(norm_answer, content, tokenizer):
    ##########single positive document encoder==========================================================================
    p_title, p_doc_sents, p_doc_weight, supp_sent_flags, _, supp_sent_labels = content
    p_doc_encode_ids, sent_start_end_pair, p_doc_len, p_title_len = document_encoder(title=p_title,
                                                                                       doc_sents=p_doc_sents,
                                                                                       tokenizer=tokenizer)
    assert len(p_doc_sents) == len(sent_start_end_pair)
    ctx_with_answer = False
    answer_positions = []  ## answer position
    for sup_sent_idx, supp_sent_flag in supp_sent_flags:
        if supp_sent_flag:
            start_id, end_id = sent_start_end_pair[sup_sent_idx]
            supp_sent_encode_ids = p_doc_encode_ids[start_id:(end_id + 1)]
            ####################################################################################################
            answer_encode_ids = tokenizer.encode(text=norm_answer, add_special_tokens=False)
            answer_start_idx = sub_list_finder(target=answer_encode_ids, source=supp_sent_encode_ids)
            if answer_start_idx < 0:
                answer_encode_ids = tokenizer.encode(text=norm_answer.strip(), add_special_tokens=False)
                answer_start_idx = sub_list_finder(target=answer_encode_ids, source=supp_sent_encode_ids)
            answer_len = len(answer_encode_ids)
            ####################################################################################################
            assert answer_start_idx >= 0, "supp sent {} \n answer={} \n p_doc = {} \n answer={} \n {} \n {}".format(
                tokenizer.decode(supp_sent_encode_ids),
                tokenizer.decode(answer_encode_ids), p_doc_sents[sup_sent_idx], norm_answer,
                supp_sent_encode_ids, answer_encode_ids)
            ctx_with_answer = True  ## support sentence with answer
            answer_positions.append((sup_sent_idx, answer_start_idx, answer_start_idx + answer_len - 1))  # supp sent idx, relative start, relative end
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if ctx_with_answer:
        pos_type = 2
    else:
        pos_type = 1
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    p_tuple = (p_doc_encode_ids, p_doc_weight, p_doc_len, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, p_title_len, pos_type)
    return p_tuple
########################################################################################################################
def neg_context_encoder(content, tokenizer):
    n_title, n_doc_sents, n_doc_weight, _, _, supp_sent_labels = content
    n_doc_encode_ids, sent_start_end_pair, n_doc_len, n_title_len = document_encoder(title=n_title,
                                                                                     doc_sents=n_doc_sents,
                                                                                     tokenizer=tokenizer)
    assert len(n_doc_sents) == len(sent_start_end_pair)
    n_tuple = (n_doc_encode_ids, n_doc_weight, n_doc_len, sent_start_end_pair, supp_sent_labels, False, [], n_title_len, 0)
    return n_tuple
########################################################################################################################
def context_encoder(content, tokenizer):
    title, doc_sents = content
    doc_encode_ids, sent_start_end_pair, doc_len, title_len = document_encoder(title=title,
                                                                                     doc_sents=doc_sents,
                                                                                     tokenizer=tokenizer)
    assert len(doc_sents) == len(sent_start_end_pair)
    n_tuple = (doc_encode_ids, doc_len, sent_start_end_pair, title_len)
    return n_tuple
########################################################################################################################
def context_random_selection(context_tuple_list, neg_num=2, shuffle=False):
    if len(context_tuple_list) <= (2 + neg_num):
        return context_tuple_list
    neg_idxes = []
    pos_idxes = []
    for doc_idx, doc_tup in enumerate(context_tuple_list):
        doc_encode_ids, doc_weight, doc_len, sent_start_end_pair, supp_sent_labels, \
        ctx_with_answer, answer_positions, title_len, pos_doc_type = doc_tup
        if pos_doc_type > 0:
            pos_idxes.append(doc_idx)
        else:
            neg_idxes.append(doc_idx)
    if neg_num > 0:
        neg_samp_idxes = np.random.choice(neg_idxes, size=neg_num)
        neg_samp_idxes = neg_samp_idxes.tolist()
        sample_idxes = pos_idxes + neg_samp_idxes
    else:
        sample_idxes = pos_idxes
    if shuffle:
        random.shuffle(sample_idxes)
    else:
        sample_idxes.sort()
    selected_context_tuple_list = [context_tuple_list[x] for x in sample_idxes]
    return selected_context_tuple_list
########################################################################################################################
def context_merge_longer(query_encode_ids, context_tuple_list, span_flag):
    ################################################################################################################
    concat_encode = query_encode_ids  ## concat encode ids (query + 10 documents)
    query_len = len(query_encode_ids)
    concat_len = query_len  ## concat encode ids length (query + 10 documents)
    ctx_len_list = []
    ################################################################################################################
    total_sent_num = 0  # compute the number of sentences
    sent_num_in_docs = [] # the number of sentences in each document = number of document
    doc_start_end_pair_list = []
    sent_start_end_pair_list = []
    sent_len_in_docs = []  ## the number of token of each sentence
    ################################################################################################################
    abs_sent2doc_map_list = []  ## length is equal to sent numbers
    abs_sentIndoc_idx_list = []  ## length is equal to sent numbers
    ################################################################################################################
    supp_sent_labels_list = []  # equal to number of sentences
    answer_position_list = []
    previous_len = query_len
    previous_sent_num = 0
    ################################################################################################################
    ctx_title_lens = []
    doc_label_list = []
    doc_label_ans_list = []
    ################################################################################################################
    for doc_idx, doc_tup in enumerate(context_tuple_list):
        doc_encode_ids, doc_weight, doc_len, sent_start_end_pair, supp_sent_labels, \
        ctx_with_answer, answer_positions, title_len, pos_doc_type = doc_tup
        ############################################################################################################
        ctx_len_list.append(doc_len)
        ############################################################################################################
        if pos_doc_type > 0:
            doc_label_list.append(pos_doc_type)
            doc_label_ans_list.append(1)
        else:
            doc_label_list.append(0)
            doc_label_ans_list.append(0)
        ############################################################################################################
        total_sent_num = total_sent_num + len(sent_start_end_pair)
        sent_num_in_docs.append(len(sent_start_end_pair))  ## number of sentences
        ############################################################################################################
        abs_sent2doc_map_list = abs_sent2doc_map_list + [doc_idx] * len(sent_start_end_pair)  ## sentence to doc idx
        abs_sentIndoc_idx_list = abs_sentIndoc_idx_list + [x for x in range(
            len(sent_start_end_pair))]  ## sentence to original sent index
        ############################################################################################################
        sent_len_in_docs = sent_len_in_docs + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
        ############################################################################################################
        ctx_title_lens.append(title_len)
        ############################################################################################################
        assert len(doc_encode_ids) == doc_len and len(sent_start_end_pair) == len(supp_sent_labels)
        concat_encode = concat_encode + doc_encode_ids
        concat_len = concat_len + doc_len
        ############################################################################################################
        # ==========================================================================================================
        doc_start_end_pair_list.append((previous_len, previous_len + doc_len - 1))
        sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
        sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
        # ==========================================================================================================
        supp_sent_labels_list = supp_sent_labels_list + supp_sent_labels
        # ==========================================================================================================
        if len(answer_positions) > 0:
            for a_idx, answer_pos in enumerate(answer_positions):
                sent_a_idx, a_start, a_end = answer_pos
                sent_off_set = sent_start_end_pair_i[sent_a_idx][0]
                temp_position = (sent_off_set + a_start, sent_off_set + a_end, sent_a_idx, doc_idx, previous_sent_num)
                answer_position_list.append(temp_position)
        previous_len = previous_len + doc_len
        previous_sent_num = previous_sent_num + len(sent_start_end_pair)
        ############################################################################################################
    ################################################################################################################
    token2sentID_map = np.zeros(concat_len).astype('int')
    for sent_idx, sent_position in enumerate(sent_start_end_pair_list):
        assert (sent_position[1] - sent_position[0] + 1) == sent_len_in_docs[sent_idx]
        token2sentID_map[sent_position[0]:(sent_position[1] + 1)] = sent_idx
    ################################################################################################################
    supp_sent_labels_ans_list = []
    for x in supp_sent_labels_list:
        if x > 0:
            supp_sent_labels_ans_list.append(1)
        else:
            supp_sent_labels_ans_list.append(0)
    ################################################################################################################
    if span_flag:
        answer_pos_start, answer_pos_end = np.array([answer_position_list[0][0]]).astype('int'), \
                                           np.array([answer_position_list[0][1]]).astype('int')
        #########################################################################################
        assert doc_label_list[answer_position_list[0][3]] > 1
        doc_label_ans_list[answer_position_list[0][3]] = 2
        assert supp_sent_labels_list[(answer_position_list[0][4] + answer_position_list[0][2])] > 1
        supp_sent_labels_ans_list[(answer_position_list[0][4] + answer_position_list[0][2])] = 2
        #########################################################################################
    else:
        answer_pos_start, answer_pos_end = np.array([0]).astype('int'), np.array([0]).astype('int')
    ################################################################################################################
    data_type = np.dtype('int,int,int,int,int')
    answer_position_tuple = np.array(answer_position_list, dtype=data_type)
    ################################################################################################################
    concat_ctx_array = np.array(concat_encode).astype('int')
    global_attn_marker = np.zeros(concat_len).astype('int')
    global_atten_marker_idxs = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                  sent_start_end_pair_list]
    global_attn_marker[global_atten_marker_idxs] = 1
    ################################################################################################################
    answer_mask = np.zeros(concat_len).astype('int')
    answer_mask_idxs = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                  sent_start_end_pair_list]
    title_mask_idxes = []  ## answer did not appear in title
    for t_idx, ti_len in enumerate(ctx_title_lens):
        if ti_len > 0:
            title_start_idx = doc_start_end_pair_list[t_idx][0]
            for j in range(ti_len):
                title_mask_idxes.append(title_start_idx + j)
    answer_mask_idxs = answer_mask_idxs + title_mask_idxes
    answer_mask[answer_mask_idxs] = 1
    ################################################################################################################
    ################################################################################################################
    doc_labels = np.array(doc_label_list).astype('int')
    doc_ans_labels = np.array(doc_label_ans_list).astype('int')
    doc_num = len(doc_label_list)
    doc_len_array = np.array(ctx_len_list).astype('int')
    sent_labels = np.array(supp_sent_labels_list).astype('int')
    sent_ans_labels = np.array(supp_sent_labels_ans_list).astype('int')
    sent_len_array = np.array(sent_len_in_docs).astype('int')
    sent_num = len(supp_sent_labels_list)
    token_num = concat_len
    sent2doc_map_array = np.array(abs_sent2doc_map_list).astype('int')
    abs_sentIndoc_array = np.array(abs_sentIndoc_idx_list).astype('int')
    ################################################################################################################
    doc_start_position = np.array([_[0] for _ in doc_start_end_pair_list]).astype('int')
    doc_end_position = np.array([_[1] for _ in doc_start_end_pair_list]).astype('int')
    ################################################################################################################
    sent_start_position = np.array([_[0] for _ in sent_start_end_pair_list]).astype('int')
    sent_end_position = np.array([_[1] for _ in sent_start_end_pair_list]).astype('int')
    ################################################################################################################
    ################################################################################################################
    doc_sent_nums = np.array(sent_num_in_docs).astype('int')
    ################################################################################################################
    doc_label_types = [x for x in enumerate(doc_label_ans_list) if x[1] > 0]
    # print(doc_label_types)
    assert len(doc_label_types) == 2
    if doc_label_types[0][1] > doc_label_types[1][1]:
        doc_head_idx, doc_tail_idx = np.array([doc_label_types[0][0]]).astype('int'), np.array([doc_label_types[1][0]]).astype('int')
    else:
        doc_head_idx, doc_tail_idx = np.array([doc_label_types[1][0]]).astype('int'), np.array([doc_label_types[0][0]]).astype('int')
    ################################################################################################################
    doc_infor_tuple = (doc_labels, doc_ans_labels, doc_num, doc_len_array, doc_start_position, doc_end_position, doc_head_idx, doc_tail_idx)
    sent_infor_tuple = (sent_labels, sent_ans_labels, sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums)
    token_infor_tuple = (concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask)
    answer_infor_tuple = (answer_pos_start, answer_pos_end, answer_position_tuple)
    ################################################################################################################
    return doc_infor_tuple, sent_infor_tuple, token_infor_tuple, answer_infor_tuple

def dev_context_merge_longer(query_encode_ids, context_tuple_list, span_flag):
    ################################################################################################################
    concat_encode = query_encode_ids  ## concat encode ids (query + 10 documents)
    query_len = len(query_encode_ids)
    concat_len = query_len  ## concat encode ids length (query + 10 documents)
    ctx_len_list = []
    ################################################################################################################
    total_sent_num = 0  # compute the number of sentences
    sent_num_in_docs = [] # the number of sentences in each document = number of document
    doc_start_end_pair_list = []
    sent_start_end_pair_list = []
    sent_len_in_docs = []  ## the number of token of each sentence
    ################################################################################################################
    abs_sent2doc_map_list = []  ## length is equal to sent numbers
    abs_sentIndoc_idx_list = []  ## length is equal to sent numbers
    ################################################################################################################
    supp_sent_labels_list = []  # equal to number of sentences
    answer_position_list = []
    previous_len = query_len
    previous_sent_num = 0
    ################################################################################################################
    ctx_title_lens = []
    doc_label_list = []
    doc_label_ans_list = []
    ################################################################################################################
    for doc_idx, doc_tup in enumerate(context_tuple_list):
        doc_encode_ids, doc_weight, doc_len, sent_start_end_pair, supp_sent_labels, \
        ctx_with_answer, answer_positions, title_len, pos_doc_type = doc_tup
        ############################################################################################################
        ctx_len_list.append(doc_len)
        ############################################################################################################
        if pos_doc_type > 0:
            doc_label_list.append(pos_doc_type)
            doc_label_ans_list.append(1)
        else:
            doc_label_list.append(0)
            doc_label_ans_list.append(0)
        ############################################################################################################
        total_sent_num = total_sent_num + len(sent_start_end_pair)
        sent_num_in_docs.append(len(sent_start_end_pair))  ## number of sentences
        ############################################################################################################
        abs_sent2doc_map_list = abs_sent2doc_map_list + [doc_idx] * len(sent_start_end_pair)  ## sentence to doc idx
        abs_sentIndoc_idx_list = abs_sentIndoc_idx_list + [x for x in range(
            len(sent_start_end_pair))]  ## sentence to original sent index
        ############################################################################################################
        sent_len_in_docs = sent_len_in_docs + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
        ############################################################################################################
        ctx_title_lens.append(title_len)
        ############################################################################################################
        assert len(doc_encode_ids) == doc_len and len(sent_start_end_pair) == len(supp_sent_labels)
        concat_encode = concat_encode + doc_encode_ids
        concat_len = concat_len + doc_len
        ############################################################################################################
        # ==========================================================================================================
        doc_start_end_pair_list.append((previous_len, previous_len + doc_len - 1))
        sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
        sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
        # ==========================================================================================================
        supp_sent_labels_list = supp_sent_labels_list + supp_sent_labels
        # ==========================================================================================================
        if len(answer_positions) > 0:
            for a_idx, answer_pos in enumerate(answer_positions):
                sent_a_idx, a_start, a_end = answer_pos
                sent_off_set = sent_start_end_pair_i[sent_a_idx][0]
                temp_position = (sent_off_set + a_start, sent_off_set + a_end, sent_a_idx, doc_idx, previous_sent_num)
                answer_position_list.append(temp_position)
        previous_len = previous_len + doc_len
        previous_sent_num = previous_sent_num + len(sent_start_end_pair)
        ############################################################################################################
    ################################################################################################################
    token2sentID_map = np.zeros(concat_len).astype('int')
    for sent_idx, sent_position in enumerate(sent_start_end_pair_list):
        assert (sent_position[1] - sent_position[0] + 1) == sent_len_in_docs[sent_idx]
        token2sentID_map[sent_position[0]:(sent_position[1] + 1)] = sent_idx
    ################################################################################################################
    supp_sent_labels_ans_list = []
    for x in supp_sent_labels_list:
        if x > 0:
            supp_sent_labels_ans_list.append(1)
        else:
            supp_sent_labels_ans_list.append(0)
    ################################################################################################################
    if span_flag:
        answer_pos_start, answer_pos_end = np.array([answer_position_list[0][0]]).astype('int'), \
                                           np.array([answer_position_list[0][1]]).astype('int')
        #########################################################################################
        assert doc_label_list[answer_position_list[0][3]] > 1
        doc_label_ans_list[answer_position_list[0][3]] = 2
        assert supp_sent_labels_list[(answer_position_list[0][4] + answer_position_list[0][2])] > 1
        supp_sent_labels_ans_list[(answer_position_list[0][4] + answer_position_list[0][2])] = 2
        #########################################################################################
    else:
        answer_pos_start, answer_pos_end = np.array([0]).astype('int'), np.array([0]).astype('int')
    ################################################################################################################
    data_type = np.dtype('int,int,int,int,int')
    answer_position_tuple = np.array(answer_position_list, dtype=data_type)
    ################################################################################################################
    concat_ctx_array = np.array(concat_encode).astype('int')
    global_attn_marker = np.zeros(concat_len).astype('int')
    global_atten_marker_idxs = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                  sent_start_end_pair_list]
    global_attn_marker[global_atten_marker_idxs] = 1
    ################################################################################################################
    answer_mask = np.zeros(concat_len).astype('int')
    answer_mask_idxs = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                  sent_start_end_pair_list]
    title_mask_idxes = []  ## answer did not appear in title
    for t_idx, ti_len in enumerate(ctx_title_lens):
        if ti_len > 0:
            title_start_idx = doc_start_end_pair_list[t_idx][0]
            for j in range(ti_len):
                title_mask_idxes.append(title_start_idx + j)
    answer_mask_idxs = answer_mask_idxs + title_mask_idxes
    answer_mask[answer_mask_idxs] = 1
    ################################################################################################################
    ################################################################################################################
    doc_labels = np.array(doc_label_list).astype('int')
    doc_ans_labels = np.array(doc_label_ans_list).astype('int')
    doc_num = len(doc_label_list)
    doc_len_array = np.array(ctx_len_list).astype('int')
    sent_labels = np.array(supp_sent_labels_list).astype('int')
    sent_ans_labels = np.array(supp_sent_labels_ans_list).astype('int')
    sent_len_array = np.array(sent_len_in_docs).astype('int')
    sent_num = len(supp_sent_labels_list)
    token_num = concat_len
    sent2doc_map_array = np.array(abs_sent2doc_map_list).astype('int')
    abs_sentIndoc_array = np.array(abs_sentIndoc_idx_list).astype('int')
    ################################################################################################################
    doc_start_position = np.array([_[0] for _ in doc_start_end_pair_list]).astype('int')
    doc_end_position = np.array([_[1] for _ in doc_start_end_pair_list]).astype('int')
    ################################################################################################################
    sent_start_position = np.array([_[0] for _ in sent_start_end_pair_list]).astype('int')
    sent_end_position = np.array([_[1] for _ in sent_start_end_pair_list]).astype('int')
    ################################################################################################################
    ################################################################################################################
    doc_sent_nums = np.array(sent_num_in_docs).astype('int')
    ################################################################################################################
    doc_label_types = [x for x in enumerate(doc_label_ans_list) if x[1] > 0]
    if len(doc_label_types) == 2:
        if doc_label_types[0][1] > doc_label_types[1][1]:
            doc_head_idx, doc_tail_idx = np.array([doc_label_types[0][0]]).astype('int'), np.array([doc_label_types[1][0]]).astype('int')
        else:
            doc_head_idx, doc_tail_idx = np.array([doc_label_types[1][0]]).astype('int'), np.array([doc_label_types[0][0]]).astype('int')
    elif len(doc_label_types) == 1:
        doc_head_idx, doc_tail_idx = np.array([0]).astype('int'), np.array([doc_label_types[0][0]]).astype('int')
    else:
        doc_head_idx, doc_tail_idx = np.array([0]).astype('int'), np.array([0]).astype('int')
    ################################################################################################################
    doc_infor_tuple = (doc_labels, doc_ans_labels, doc_num, doc_len_array, doc_start_position, doc_end_position, doc_head_idx, doc_tail_idx)
    sent_infor_tuple = (sent_labels, sent_ans_labels, sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums)
    token_infor_tuple = (concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask)
    answer_infor_tuple = (answer_pos_start, answer_pos_end, answer_position_tuple)
    ################################################################################################################
    return doc_infor_tuple, sent_infor_tuple, token_infor_tuple, answer_infor_tuple

def test_context_merge_longer(query_encode_ids, context_tuple_list):
    ################################################################################################################
    concat_encode = query_encode_ids  ## concat encode ids (query + 10 documents)
    query_len = len(query_encode_ids)
    concat_len = query_len  ## concat encode ids length (query + 10 documents)
    ctx_len_list = []
    ################################################################################################################
    total_sent_num = 0  # compute the number of sentences
    sent_num_in_docs = [] # the number of sentences in each document = number of document
    doc_start_end_pair_list = []
    sent_start_end_pair_list = []
    sent_len_in_docs = []  ## the number of token of each sentence
    ################################################################################################################
    abs_sent2doc_map_list = []  ## length is equal to sent numbers
    abs_sentIndoc_idx_list = []  ## length is equal to sent numbers
    ################################################################################################################
    previous_len = query_len
    ################################################################################################################
    ctx_title_lens = []
    ################################################################################################################
    ################################################################################################################
    for doc_idx, doc_tup in enumerate(context_tuple_list):
        doc_encode_ids, doc_len, sent_start_end_pair, title_len = doc_tup
        ############################################################################################################
        ctx_len_list.append(doc_len)
        ############################################################################################################
        total_sent_num = total_sent_num + len(sent_start_end_pair)
        sent_num_in_docs.append(len(sent_start_end_pair))  ## number of sentences
        ############################################################################################################
        abs_sent2doc_map_list = abs_sent2doc_map_list + [doc_idx] * len(sent_start_end_pair)  ## sentence to doc idx
        abs_sentIndoc_idx_list = abs_sentIndoc_idx_list + [x for x in range(
            len(sent_start_end_pair))]  ## sentence to original sent index
        ############################################################################################################
        sent_len_in_docs = sent_len_in_docs + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
        ############################################################################################################
        ctx_title_lens.append(title_len)
        ############################################################################################################
        assert len(doc_encode_ids) == doc_len
        concat_encode = concat_encode + doc_encode_ids
        concat_len = concat_len + doc_len
        ############################################################################################################
        # ==========================================================================================================
        doc_start_end_pair_list.append((previous_len, previous_len + doc_len - 1))
        sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
        sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
        # ==========================================================================================================
        previous_len = previous_len + doc_len
        ############################################################################################################
    ################################################################################################################
    token2sentID_map = np.zeros(concat_len).astype('int')
    for sent_idx, sent_position in enumerate(sent_start_end_pair_list):
        assert (sent_position[1] - sent_position[0] + 1) == sent_len_in_docs[sent_idx]
        token2sentID_map[sent_position[0]:(sent_position[1] + 1)] = sent_idx
    ################################################################################################################
    ################################################################################################################
    concat_ctx_array = np.array(concat_encode).astype('int')
    global_attn_marker = np.zeros(concat_len).astype('int')
    global_atten_marker_idxs = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                  sent_start_end_pair_list]
    global_attn_marker[global_atten_marker_idxs] = 1
    ################################################################################################################
    answer_mask = np.zeros(concat_len).astype('int')
    answer_mask_idxs = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                  sent_start_end_pair_list]
    title_mask_idxes = []  ## answer did not appear in title
    for t_idx, ti_len in enumerate(ctx_title_lens):
        if ti_len > 0:
            title_start_idx = doc_start_end_pair_list[t_idx][0]
            for j in range(ti_len):
                title_mask_idxes.append(title_start_idx + j)
    answer_mask_idxs = answer_mask_idxs + title_mask_idxes
    answer_mask[answer_mask_idxs] = 1
    ################################################################################################################
    ################################################################################################################
    doc_num = len(doc_start_end_pair_list)
    doc_len_array = np.array(ctx_len_list).astype('int')
    sent_len_array = np.array(sent_len_in_docs).astype('int')
    sent_num = len(sent_start_end_pair_list)
    token_num = concat_len
    sent2doc_map_array = np.array(abs_sent2doc_map_list).astype('int')
    abs_sentIndoc_array = np.array(abs_sentIndoc_idx_list).astype('int')
    ################################################################################################################
    doc_start_position = np.array([_[0] for _ in doc_start_end_pair_list]).astype('int')
    doc_end_position = np.array([_[1] for _ in doc_start_end_pair_list]).astype('int')
    ################################################################################################################
    sent_start_position = np.array([_[0] for _ in sent_start_end_pair_list]).astype('int')
    sent_end_position = np.array([_[1] for _ in sent_start_end_pair_list]).astype('int')
    ################################################################################################################
    doc_sent_nums = np.array(sent_num_in_docs).astype('int')
    ################################################################################################################
    doc_infor_tuple = (doc_num, doc_len_array, doc_start_position, doc_end_position)
    sent_infor_tuple = (sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums)
    token_infor_tuple = (concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask)
    ################################################################################################################
    return doc_infor_tuple, sent_infor_tuple, token_infor_tuple

########################################################################################################################
def mask_generation(sent_num_docs: list, max_sent_num: int):
    assert len(sent_num_docs) > 0 and sent_num_docs[0] > 0
    ss_attn_mask = np.ones((sent_num_docs[0], sent_num_docs[0]))
    sd_attn_mask = np.ones((1, sent_num_docs[0]))
    doc_pad_num = 0
    for idx in range(1, len(sent_num_docs)):
        sent_num_i = sent_num_docs[idx]
        if sent_num_i > 0:
            ss_mask_i = np.ones((sent_num_i, sent_num_i))
            ss_attn_mask = block_diag(ss_attn_mask, ss_mask_i)
            sd_mask_i = np.ones((1, sent_num_i))
            sd_attn_mask = block_diag(sd_attn_mask, sd_mask_i)
        else:
            doc_pad_num = doc_pad_num + 1

    sent_num_sum = sum(sent_num_docs)
    assert sent_num_sum <= max_sent_num, '{}, max {}'.format(sent_num_sum, max_sent_num)
    ss_attn_mask = torch.from_numpy(ss_attn_mask).type(torch.bool)
    sd_attn_mask = torch.from_numpy(sd_attn_mask).type(torch.bool)
    sent_pad_num = max_sent_num - sent_num_sum
    if sent_pad_num > 0:
        ss_attn_mask = F.pad(ss_attn_mask, [0, sent_pad_num, 0, sent_pad_num], 'constant', False)
        sd_attn_mask = F.pad(sd_attn_mask, [0, sent_pad_num, 0, 0], 'constant', False)
    if doc_pad_num > 0:
        sd_attn_mask = F.pad(sd_attn_mask, [0, 0, 0, doc_pad_num], 'constant', False)
    return ss_attn_mask, sd_attn_mask