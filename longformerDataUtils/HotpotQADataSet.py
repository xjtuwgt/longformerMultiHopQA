import torch
import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import LongformerTokenizer
from longformerDataUtils.hotpotQAUtils import mask_generation
from longformerDataUtils.hotpotQAUtils import context_random_selection, context_merge_longer, dev_context_merge_longer, test_context_merge_longer
####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HotpotTrainDataset(Dataset): ##for training data loader
    def __init__(self, data_frame: DataFrame, tokenizer: LongformerTokenizer, max_token_num=4096, max_doc_num=10, max_sent_num=150,
                 neg_drop_out=0.1,
                 shuffle=False):
        assert max_doc_num >= 2
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_token_num = max_token_num
        self.max_doc_num = max_doc_num
        self.neg_doc_num = max_doc_num - 2
        self.max_sent_num = max_sent_num
        self.pad_token_id = tokenizer.pad_token_id
        self.neg_drop_out = neg_drop_out
        self.shuffle = shuffle

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        example_id = torch.LongTensor([example['e_id']])  ### for ddp validation alignment
        ####
        query_encode, ctx_encode_list, norm_answer = example['ques_encode'], example['ctx_encode_list'], example['norm_answer']
        ctx_encode = context_random_selection(context_tuple_list=ctx_encode_list, neg_num=self.neg_doc_num, shuffle=self.shuffle)
        if norm_answer.strip() in ['yes', 'no', 'noanswer']:  ## yes: 1, no/noanswer: 2, span = 0
            answer_type_label = np.array([1]) if norm_answer.strip() == 'yes' else np.array([2])
        else:
            answer_type_label = np.array([0])
        span_flag = example['span_flag']
        doc_infor, sent_infor, seq_infor, answer_infor = context_merge_longer(query_encode_ids=query_encode,
                                                                              context_tuple_list=ctx_encode,
                                                                              span_flag=span_flag)
        doc_labels, doc_ans_labels, doc_num, doc_len_array, doc_start_position, doc_end_position, doc_head_idx, doc_tail_idx = doc_infor
        sent_labels, sent_ans_labels, sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums = sent_infor
        concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask_idxs = seq_infor
        answer_pos_start, answer_pos_end, _ = answer_infor
        ####
        doc_num = doc_num
        concat_sent_num = sent_num
        concat_len = token_num
        ####
        cat_doc_encodes = concat_ctx_array.tolist()
        cat_doc_attention_mask = [1] * concat_len
        cat_doc_global_attn_mask = global_attn_marker.tolist()
        ctx_marker_mask = answer_mask_idxs.tolist()
        ctx_token2sent_map = token2sentID_map.tolist()
        assert concat_len == len(cat_doc_encodes)

        if concat_len < self.max_token_num:
            token_pad_num = self.max_token_num - concat_len
            cat_doc_encodes = cat_doc_encodes + [self.pad_token_id] * token_pad_num
            cat_doc_attention_mask = cat_doc_attention_mask + [0] * token_pad_num
            ctx_token2sent_map = ctx_token2sent_map + [0] * token_pad_num
            cat_doc_global_attn_mask = cat_doc_global_attn_mask + [0] * token_pad_num
            ctx_marker_mask = ctx_marker_mask + [0] * token_pad_num
        cat_doc_encodes = torch.LongTensor(cat_doc_encodes)
        cat_doc_attention_mask = torch.LongTensor(cat_doc_attention_mask)
        ctx_token2sent_map = torch.LongTensor(ctx_token2sent_map)
        cat_doc_global_attn_mask = torch.LongTensor(cat_doc_global_attn_mask)
        ctx_marker_mask = torch.BoolTensor(ctx_marker_mask)
        ################################################################################################################
        doc_start_idxes = doc_start_position.tolist()
        doc_end_idxes = doc_end_position.tolist()
        doc_lens = doc_len_array.tolist()
        doc_labels = doc_ans_labels.tolist()
        doc_sent_nums = doc_sent_nums.tolist()
        if doc_num < self.max_doc_num:
            doc_pad_num = self.max_doc_num - doc_num
            doc_start_idxes = doc_start_idxes + [0] * doc_pad_num
            doc_end_idxes = doc_end_idxes + [0] * doc_pad_num
            doc_lens = doc_lens + [0] * doc_pad_num
            doc_labels = doc_labels + [0] * doc_pad_num
            doc_sent_nums = doc_sent_nums + [0] * doc_pad_num
        doc_start_idxes = torch.LongTensor(doc_start_idxes)
        doc_end_idxes = torch.LongTensor(doc_end_idxes)
        doc_lens = torch.LongTensor(doc_lens)
        doc_labels = torch.LongTensor(doc_labels)
        ################################################################################################################
        sent_start_idxes = sent_start_position.tolist()
        sent_end_idxes = sent_end_position.tolist()
        ctx_sent_lens = sent_len_array.tolist()
        supp_sent_labels = sent_ans_labels.tolist()
        ctx_sent2doc_map = sent2doc_map_array.tolist()
        ctx_sentIndoc_idx = abs_sentIndoc_array.tolist()
        if concat_sent_num < self.max_sent_num:
            sent_pad_num = self.max_sent_num - concat_sent_num
            sent_start_idxes = sent_start_idxes + [0] * sent_pad_num
            sent_end_idxes = sent_end_idxes + [0] * sent_pad_num
            ctx_sent_lens = ctx_sent_lens + [0] * sent_pad_num
            supp_sent_labels = supp_sent_labels + [0] * sent_pad_num
            ctx_sent2doc_map = ctx_sent2doc_map + [0] * sent_pad_num
            ctx_sentIndoc_idx = ctx_sentIndoc_idx + [0] * sent_pad_num
        sent_start_idxes = torch.LongTensor(sent_start_idxes)
        sent_end_idxes = torch.LongTensor(sent_end_idxes)
        ctx_sent_lens = torch.LongTensor(ctx_sent_lens)
        supp_sent_labels = torch.LongTensor(supp_sent_labels)
        ctx_sent2doc_map = torch.LongTensor(ctx_sent2doc_map)
        ctx_sentIndoc_idx = torch.LongTensor(ctx_sentIndoc_idx)
        ################################################################################################################
        answer_start_idx = answer_pos_start
        answer_end_idx = answer_pos_end
        answer_start_idx = torch.LongTensor(answer_start_idx)
        answer_end_idx = torch.LongTensor(answer_end_idx)
        ################################################################################################################
        yes_no_label = answer_type_label
        yes_no_label = torch.LongTensor(yes_no_label)
        ################################################################################################################
        head_doc_idx = doc_head_idx
        tail_doc_idx = doc_tail_idx
        head_doc_idx = torch.LongTensor(head_doc_idx)
        tail_doc_idx = torch.LongTensor(tail_doc_idx)
        ################################################################################################################
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=doc_sent_nums, max_sent_num=self.max_sent_num)
        ################################################################################################################
        ################################################################################################################
        assert concat_len <= self.max_token_num
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, sent_start_idxes, \
               answer_start_idx, answer_end_idx, doc_lens, doc_labels, ctx_sent_lens, supp_sent_labels, yes_no_label, head_doc_idx, \
               tail_doc_idx, ss_attn_mask, sd_attn_mask, ctx_sent2doc_map, ctx_sentIndoc_idx, ctx_token2sent_map, ctx_marker_mask, \
               doc_end_idxes, sent_end_idxes, concat_len, concat_sent_num, example_id

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[22] for _ in data])
        batch_max_sent_num = max([_[23] for _ in data])
        batch_ids = torch.stack([_[24] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[4] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_answer_starts = torch.stack([_[5] for _ in data], dim=0)
        batch_answer_ends = torch.stack([_[6] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[7] for _ in data], dim=0)
        batch_doc_labels = torch.stack([_[8] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[9] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        batch_sent_labels = torch.stack([_[10] for _ in data], dim=0)
        batch_sent_labels = batch_sent_labels[:, range(0, batch_max_sent_num)]
        batch_yes_no = torch.stack([_[11] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_head_idx = torch.stack([_[12] for _ in data], dim=0)
        batch_tail_idx = torch.stack([_[13] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[14] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[15] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent2doc_map = torch.stack([_[16] for _ in data], dim=0)
        batch_sent2doc_map = batch_sent2doc_map[:, range(0, batch_max_sent_num)]
        batch_sentIndoc_map = torch.stack([_[17] for _ in data], dim=0)
        batch_sentIndoc_map = batch_sentIndoc_map[:, range(0, batch_max_sent_num)]
        batch_token2sent = torch.stack([_[18] for _ in data], dim=0)
        batch_token2sent = batch_token2sent[:, range(0, batch_max_ctx_len)]
        batch_marker = torch.stack([_[19] for _ in data], dim=0)
        batch_marker = batch_marker[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_ends = torch.stack([_[20] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_ends = torch.stack([_[21] for _ in data], dim=0)
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts, 'doc_end': batch_doc_ends,
                'sent_start': batch_sent_starts, 'sent_end': batch_sent_ends, 'ans_start': batch_answer_starts, 'ans_end': batch_answer_ends,
               'doc_lens': batch_doc_lens, 'doc_labels': batch_doc_labels, 'sent_lens': batch_sent_lens,
               'sent_labels': batch_sent_labels, 'yes_no': batch_yes_no, 'head_idx': batch_head_idx,
               'tail_idx': batch_tail_idx, 'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask, 'id': batch_ids,
               's2d_map': batch_sent2doc_map, 'sInd_map': batch_sentIndoc_map, 'marker': batch_marker, 't2s_map': batch_token2sent}
        return res
##**********************************************************************************************************************
class HotpotDevDataset(Dataset): ##for dev dataloader
    def __init__(self, data_frame: DataFrame, tokenizer: LongformerTokenizer, max_token_num=4096, max_doc_num=10,
                 max_sent_num=150):
        assert max_doc_num >= 2
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_token_num = max_token_num
        self.max_doc_num = max_doc_num
        self.neg_doc_num = max_doc_num - 2
        self.max_sent_num = max_sent_num
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        example_id = torch.LongTensor([example['e_id']])  ### for ddp validation alignment
        ####
        query_encode, ctx_encode_list, norm_answer = example['ques_encode'], example['ctx_encode_list'], example['norm_answer']
        if norm_answer.strip() in ['yes', 'no', 'noanswer']:  ## yes: 1, no/noanswer: 2, span = 0
            answer_type_label = np.array([1]) if norm_answer.strip() == 'yes' else np.array([2])
        else:
            answer_type_label = np.array([0])
        span_flag = example['span_flag']
        doc_infor, sent_infor, seq_infor, answer_infor = dev_context_merge_longer(query_encode_ids=query_encode,
                                                                              context_tuple_list=ctx_encode_list,
                                                                              span_flag=span_flag)
        doc_labels, doc_ans_labels, doc_num, doc_len_array, doc_start_position, doc_end_position, _, _ = doc_infor ## no head, tail infor
        sent_labels, sent_ans_labels, sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums = sent_infor
        concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask_idxs = seq_infor
        answer_pos_start, answer_pos_end, _ = answer_infor
        ####
        doc_num = doc_num
        concat_sent_num = sent_num
        concat_len = token_num
        ####
        cat_doc_encodes = concat_ctx_array.tolist()
        cat_doc_attention_mask = [1] * concat_len
        cat_doc_global_attn_mask = global_attn_marker.tolist()
        ctx_marker_mask = answer_mask_idxs.tolist()
        ctx_token2sent_map = token2sentID_map.tolist()
        assert concat_len == len(cat_doc_encodes)

        if concat_len < self.max_token_num:
            token_pad_num = self.max_token_num - concat_len
            cat_doc_encodes = cat_doc_encodes + [self.pad_token_id] * token_pad_num
            cat_doc_attention_mask = cat_doc_attention_mask + [0] * token_pad_num
            ctx_token2sent_map = ctx_token2sent_map + [0] * token_pad_num
            cat_doc_global_attn_mask = cat_doc_global_attn_mask + [0] * token_pad_num
            ctx_marker_mask = ctx_marker_mask + [0] * token_pad_num
        cat_doc_encodes = torch.LongTensor(cat_doc_encodes)
        cat_doc_attention_mask = torch.LongTensor(cat_doc_attention_mask)
        ctx_token2sent_map = torch.LongTensor(ctx_token2sent_map)
        cat_doc_global_attn_mask = torch.LongTensor(cat_doc_global_attn_mask)
        ctx_marker_mask = torch.BoolTensor(ctx_marker_mask)
        ################################################################################################################
        doc_start_idxes = doc_start_position.tolist()
        doc_end_idxes = doc_end_position.tolist()
        doc_lens = doc_len_array.tolist()
        doc_labels = doc_ans_labels.tolist()
        doc_sent_nums = doc_sent_nums.tolist()
        if doc_num < self.max_doc_num:
            doc_pad_num = self.max_doc_num - doc_num
            doc_start_idxes = doc_start_idxes + [0] * doc_pad_num
            doc_end_idxes = doc_end_idxes + [0] * doc_pad_num
            doc_lens = doc_lens + [0] * doc_pad_num
            doc_labels = doc_labels + [0] * doc_pad_num
            doc_sent_nums = doc_sent_nums + [0] * doc_pad_num
        doc_start_idxes = torch.LongTensor(doc_start_idxes)
        doc_end_idxes = torch.LongTensor(doc_end_idxes)
        doc_lens = torch.LongTensor(doc_lens)
        doc_labels = torch.LongTensor(doc_labels)
        ################################################################################################################
        sent_start_idxes = sent_start_position.tolist()
        sent_end_idxes = sent_end_position.tolist()
        ctx_sent_lens = sent_len_array.tolist()
        supp_sent_labels = sent_ans_labels.tolist()
        ctx_sent2doc_map = sent2doc_map_array.tolist()
        ctx_sentIndoc_idx = abs_sentIndoc_array.tolist()
        if concat_sent_num < self.max_sent_num:
            sent_pad_num = self.max_sent_num - concat_sent_num
            sent_start_idxes = sent_start_idxes + [0] * sent_pad_num
            sent_end_idxes = sent_end_idxes + [0] * sent_pad_num
            ctx_sent_lens = ctx_sent_lens + [0] * sent_pad_num
            supp_sent_labels = supp_sent_labels + [0] * sent_pad_num
            ctx_sent2doc_map = ctx_sent2doc_map + [0] * sent_pad_num
            ctx_sentIndoc_idx = ctx_sentIndoc_idx + [0] * sent_pad_num
        sent_start_idxes = torch.LongTensor(sent_start_idxes)
        sent_end_idxes = torch.LongTensor(sent_end_idxes)
        ctx_sent_lens = torch.LongTensor(ctx_sent_lens)
        supp_sent_labels = torch.LongTensor(supp_sent_labels)
        ctx_sent2doc_map = torch.LongTensor(ctx_sent2doc_map)
        ctx_sentIndoc_idx = torch.LongTensor(ctx_sentIndoc_idx)
        ################################################################################################################
        answer_start_idx = answer_pos_start
        answer_end_idx = answer_pos_end
        answer_start_idx = torch.LongTensor(answer_start_idx)
        answer_end_idx = torch.LongTensor(answer_end_idx)
        ################################################################################################################
        yes_no_label = answer_type_label
        yes_no_label = torch.LongTensor(yes_no_label)
        ################################################################################################################
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=doc_sent_nums, max_sent_num=self.max_sent_num)
        ################################################################################################################
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert concat_len <= self.max_token_num
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, sent_start_idxes, \
               answer_start_idx, answer_end_idx, doc_lens, doc_labels, ctx_sent_lens, supp_sent_labels, yes_no_label, \
               ss_attn_mask, sd_attn_mask, ctx_sent2doc_map, ctx_sentIndoc_idx, ctx_token2sent_map, ctx_marker_mask, \
               doc_end_idxes, sent_end_idxes, concat_len, concat_sent_num, example_id

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[20] for _ in data])
        batch_max_sent_num = max([_[21] for _ in data])
        batch_ids = torch.stack([_[22] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[4] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_answer_starts = torch.stack([_[5] for _ in data], dim=0)
        batch_answer_ends = torch.stack([_[6] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[7] for _ in data], dim=0)
        batch_doc_labels = torch.stack([_[8] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[9] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        batch_sent_labels = torch.stack([_[10] for _ in data], dim=0)
        batch_sent_labels = batch_sent_labels[:, range(0, batch_max_sent_num)]
        batch_yes_no = torch.stack([_[11] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[12] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[13] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent2doc_map = torch.stack([_[14] for _ in data], dim=0)
        batch_sent2doc_map = batch_sent2doc_map[:, range(0, batch_max_sent_num)]
        batch_sentIndoc_map = torch.stack([_[15] for _ in data], dim=0)
        batch_sentIndoc_map = batch_sentIndoc_map[:, range(0, batch_max_sent_num)]
        batch_token2sent = torch.stack([_[16] for _ in data], dim=0)
        batch_token2sent = batch_token2sent[:, range(0, batch_max_ctx_len)]
        batch_marker = torch.stack([_[17] for _ in data], dim=0)
        batch_marker = batch_marker[:, range(0, batch_max_ctx_len)]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_ends = torch.stack([_[18] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_ends = torch.stack([_[19] for _ in data], dim=0)
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts, 'doc_end': batch_doc_ends,
                'sent_start': batch_sent_starts, 'sent_end': batch_sent_ends, 'ans_start': batch_answer_starts, 'ans_end': batch_answer_ends,
               'doc_lens': batch_doc_lens, 'doc_labels': batch_doc_labels, 'sent_lens': batch_sent_lens,
               'sent_labels': batch_sent_labels, 'yes_no': batch_yes_no, 'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask,
               's2d_map': batch_sent2doc_map, 'sInd_map': batch_sentIndoc_map, 'marker': batch_marker,
               't2s_map': batch_token2sent, 'id': batch_ids}
        return res
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HotpotTestDataset(Dataset): ##for dev dataloader
    ##ques_encode', 'ques_len', 'ctx_encode', 'ctx_lens', 'ctx_max_len'
    def __init__(self, data_frame: DataFrame, tokenizer: LongformerTokenizer, max_token_num=4096, max_doc_num=2,
                 max_sent_num=150):
        assert max_doc_num >= 2
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_token_num = max_token_num
        self.max_doc_num = max_doc_num
        self.max_sent_num = max_sent_num
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        example_id = torch.LongTensor([example['e_id']])  ### for ddp validation alignment
        ####
        query_encode, ctx_encode = example['ques_encode'], example['ctx_encode_list']
        doc_infor, sent_infor, seq_infor = test_context_merge_longer(query_encode_ids=query_encode,
                                                                              context_tuple_list=ctx_encode)
        doc_num, doc_len_array, doc_start_position, doc_end_position = doc_infor
        sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums = sent_infor
        concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask_idxs = seq_infor
        ####
        doc_num = doc_num
        concat_sent_num = sent_num
        concat_len = token_num
        ####
        cat_doc_encodes = concat_ctx_array.tolist()
        cat_doc_attention_mask = [1] * concat_len
        cat_doc_global_attn_mask = global_attn_marker.tolist()
        ctx_marker_mask = answer_mask_idxs.tolist()
        ctx_token2sent_map = token2sentID_map.tolist()
        assert concat_len == len(cat_doc_encodes)

        if concat_len < self.max_token_num:
            token_pad_num = self.max_token_num - concat_len
            cat_doc_encodes = cat_doc_encodes + [self.pad_token_id] * token_pad_num
            cat_doc_attention_mask = cat_doc_attention_mask + [0] * token_pad_num
            ctx_token2sent_map = ctx_token2sent_map + [0] * token_pad_num
            cat_doc_global_attn_mask = cat_doc_global_attn_mask + [0] * token_pad_num
            ctx_marker_mask = ctx_marker_mask + [0] * token_pad_num
        cat_doc_encodes = torch.LongTensor(cat_doc_encodes)
        cat_doc_attention_mask = torch.LongTensor(cat_doc_attention_mask)
        ctx_token2sent_map = torch.LongTensor(ctx_token2sent_map)
        cat_doc_global_attn_mask = torch.LongTensor(cat_doc_global_attn_mask)
        ctx_marker_mask = torch.BoolTensor(ctx_marker_mask)
        ################################################################################################################
        doc_start_idxes = doc_start_position.tolist()
        doc_end_idxes = doc_end_position.tolist()
        doc_lens = doc_len_array.tolist()
        doc_sent_nums = doc_sent_nums.tolist()
        if doc_num < self.max_doc_num:
            doc_pad_num = self.max_doc_num - doc_num
            doc_start_idxes = doc_start_idxes + [0] * doc_pad_num
            doc_end_idxes = doc_end_idxes + [0] * doc_pad_num
            doc_lens = doc_lens + [0] * doc_pad_num
            doc_sent_nums = doc_sent_nums + [0] * doc_pad_num
        doc_start_idxes = torch.LongTensor(doc_start_idxes)
        doc_end_idxes = torch.LongTensor(doc_end_idxes)
        doc_lens = torch.LongTensor(doc_lens)
        ################################################################################################################
        sent_start_idxes = sent_start_position.tolist()
        sent_end_idxes = sent_end_position.tolist()
        ctx_sent_lens = sent_len_array.tolist()
        ctx_sent2doc_map = sent2doc_map_array.tolist()
        ctx_sentIndoc_idx = abs_sentIndoc_array.tolist()
        if concat_sent_num < self.max_sent_num:
            sent_pad_num = self.max_sent_num - concat_sent_num
            sent_start_idxes = sent_start_idxes + [0] * sent_pad_num
            sent_end_idxes = sent_end_idxes + [0] * sent_pad_num
            ctx_sent_lens = ctx_sent_lens + [0] * sent_pad_num
            ctx_sent2doc_map = ctx_sent2doc_map + [0] * sent_pad_num
            ctx_sentIndoc_idx = ctx_sentIndoc_idx + [0] * sent_pad_num
        sent_start_idxes = torch.LongTensor(sent_start_idxes)
        sent_end_idxes = torch.LongTensor(sent_end_idxes)
        ctx_sent_lens = torch.LongTensor(ctx_sent_lens)
        ctx_sent2doc_map = torch.LongTensor(ctx_sent2doc_map)
        ctx_sentIndoc_idx = torch.LongTensor(ctx_sentIndoc_idx)
        ################################################################################################################
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=doc_sent_nums, max_sent_num=self.max_sent_num)
        ################################################################################################################
        assert concat_len <= self.max_token_num
        ################################################################################################################
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, sent_start_idxes, \
               doc_lens, ctx_sent_lens, ss_attn_mask, sd_attn_mask, ctx_sent2doc_map, ctx_sentIndoc_idx, ctx_token2sent_map, ctx_marker_mask, \
               doc_end_idxes, sent_end_idxes, concat_len, concat_sent_num, example_id

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[15] for _ in data])
        batch_max_sent_num = max([_[16] for _ in data])
        batch_ids = torch.stack([_[17] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[4] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[5] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[6] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[7] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[8] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent2doc_map = torch.stack([_[9] for _ in data], dim=0)
        batch_sent2doc_map = batch_sent2doc_map[:, range(0, batch_max_sent_num)]
        batch_sentIndoc_map = torch.stack([_[10] for _ in data], dim=0)
        batch_sentIndoc_map = batch_sentIndoc_map[:, range(0, batch_max_sent_num)]
        batch_token2sent = torch.stack([_[11] for _ in data], dim=0)
        batch_token2sent = batch_token2sent[:, range(0, batch_max_ctx_len)]
        batch_marker = torch.stack([_[12] for _ in data], dim=0)
        batch_marker = batch_marker[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_ends = torch.stack([_[13] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_ends = torch.stack([_[14] for _ in data], dim=0)
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts, 'doc_end': batch_doc_ends,
               'sent_start': batch_sent_starts, 'sent_end': batch_sent_ends, 'id': batch_ids,
               'doc_lens': batch_doc_lens, 'sent_lens': batch_sent_lens, 'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask,
               's2d_map': batch_sent2doc_map, 'sInd_map': batch_sentIndoc_map, 'marker': batch_marker,
               't2s_map': batch_token2sent}
        return res