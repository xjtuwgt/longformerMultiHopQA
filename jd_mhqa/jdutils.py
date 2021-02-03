from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from eval.hotpot_doc_evaluate import doc_recall_eval
import json
import shutil
from torch import nn
from csr_mhqa.data_processing import IGNORE_INDEX
from csr_mhqa.utils import convert_to_tokens
from jd_mhqa.lossUtils import ATPLoss
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_sent_prediction(predict_support_np_ith, example_dict, batch_ids_ith, thresholds):
    N_thresh = len(thresholds)
    cur_sp_pred = [[] for _ in range(N_thresh)]
    cur_id = batch_ids_ith
    arg_order_ids = np.argsort(predict_support_np_ith)[::-1].tolist()
    filtered_arg_order_ids = [_ for _ in arg_order_ids if _ < len(example_dict[cur_id].sent_names)]
    assert len(filtered_arg_order_ids) >= 2
    for thresh_i in range(N_thresh):
        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[filtered_arg_order_ids[0]])
        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[filtered_arg_order_ids[1]])
    second_score = predict_support_np_ith[filtered_arg_order_ids[1]]
    for j in range(2, len(filtered_arg_order_ids)):
        jth_idx = filtered_arg_order_ids[j]
        for thresh_i in range(N_thresh):
            if predict_support_np_ith[jth_idx] > thresholds[thresh_i] * second_score:
                cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[jth_idx])
    return cur_sp_pred
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_sent_prediction_hgn(predict_support_np_ith, example_dict, batch_ids_ith, thresholds):
    N_thresh = len(thresholds)
    cur_sp_pred = [[] for _ in range(N_thresh)]
    cur_id = batch_ids_ith
    for j in range(predict_support_np_ith.shape[0]):
        if j >= len(example_dict[cur_id].sent_names):
            break
        for thresh_i in range(N_thresh):
            if predict_support_np_ith[j] > thresholds[thresh_i]:
                cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])
                # print(example_dict[cur_id].sent_names[j])
    return cur_sp_pred
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_doc_prediction(predict_para_support_np_ith, example_dict, batch_ids_ith):
    arg_order_ids = np.argsort(predict_para_support_np_ith)[::-1].tolist()
    cand_para_names = example_dict[batch_ids_ith].para_names
    assert len(cand_para_names) >=2
    cur_sp_para_pred = [cand_para_names[arg_order_ids[0]], cand_para_names[arg_order_ids[1]]]
    return cur_sp_para_pred
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_doc_sent_consistent_checker(predict_para_dict: dict, predicted_supp_sent_dict: dict, gold_file):
    def consistent_checker(predict_para_support, pred_titles):
        if len(pred_titles) != len(predict_para_support):
            return False
        for p_title in pred_titles:
            if p_title not in predict_para_support:
                return False
        return True

    total_inconsist_num = 0
    total_inconsist_gold_num = 0

    with open(gold_file) as f:
        gold = json.load(f)
    for dp in gold:
        para_id = dp['_id']
        supp_titles = list(set([x[0] for x in dp['supporting_facts']]))
        predict_para = predict_para_dict[para_id]
        predict_supp_sents = predicted_supp_sent_dict[para_id]
        pred_titles = list(set([x[0] for x in predict_supp_sents]))
        whether_consist = consistent_checker(predict_para_support=predict_para, pred_titles=pred_titles)
        whether_good_consist = consistent_checker(predict_para_support=predict_para, pred_titles=supp_titles)
        if not whether_consist:
            total_inconsist_num = total_inconsist_num + 1

    return total_inconsist_num
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_sent_prediction_with_para_constraint(predict_para_support_np_ith, predict_support_np_ith, example_dict, batch_ids_ith, thresholds):
    ############################################
    cur_sp_para_pred = supp_doc_prediction(predict_para_support_np_ith=predict_para_support_np_ith, example_dict=example_dict,
                                           batch_ids_ith=batch_ids_ith)
    assert len(cur_sp_para_pred) == 2
    ############################################
    N_thresh = len(thresholds)
    cur_sp_pred = [[] for _ in range(N_thresh)]
    cur_id = batch_ids_ith
    arg_order_ids = np.argsort(predict_support_np_ith)[::-1].tolist()
    filtered_arg_order_ids = [_ for _ in arg_order_ids if _ < len(example_dict[cur_id].sent_names)]
    assert len(filtered_arg_order_ids) >= 2
    for thresh_i in range(N_thresh):
        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[filtered_arg_order_ids[0]])
        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[filtered_arg_order_ids[1]])
    second_score = predict_support_np_ith[filtered_arg_order_ids[1]]
    for j in range(2, len(filtered_arg_order_ids)):
        jth_idx = filtered_arg_order_ids[j]
        for thresh_i in range(N_thresh):
            if predict_support_np_ith[jth_idx] > thresholds[thresh_i] * second_score \
                    and example_dict[cur_id].sent_names[jth_idx][0] in cur_sp_para_pred:
            # if predict_support_np_ith[jth_idx] > thresholds[thresh_i] * second_score:
                cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[jth_idx])
    ############################################
    return cur_sp_pred, cur_sp_para_pred
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def jd_eval_model(args, encoder, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file):
    encoder.eval()
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    dataloader.refresh()

    thresholds = np.arange(0.1, 1.0, 0.025)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]
    ##++++++++++++++++++++++++++++++++++
    total_para_sp_dict = {}
    ##++++++++++++++++++++++++++++++++++

    for batch in tqdm(dataloader):
        with torch.no_grad():
            inputs = {'input_ids':      batch['context_idxs'],
                      'attention_mask': batch['context_mask'],
                      'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids
            outputs = encoder(**inputs)

            batch['context_encoding'] = outputs[0]
            batch['context_mask'] = batch['context_mask'].float().to(args.device)
            start, end, q_type, paras, sent, ent, yp1, yp2 = model(batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)

        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        ##++++++++++++++++++++++++++++++++++++++++
        predict_para_support_np = torch.sigmoid(paras[:, :, 1]).data.cpu().numpy()
        ##++++++++++++++++++++++++++++++++++++++++
        # print('sent shape {}'.format(sent.shape))
        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()
        # print('supp sent np shape {}'.format(predict_support_np.shape))
        for i in range(predict_support_np.shape[0]):
            cur_id = batch['ids'][i]
            predict_para_support_np_ith = predict_para_support_np[i]
            predict_support_np_ith = predict_support_np[i]
            # ####################################
            cur_para_sp_pred = supp_doc_prediction(predict_para_support_np_ith=predict_para_support_np_ith, example_dict=example_dict, batch_ids_ith=cur_id)
            total_para_sp_dict[cur_id] = cur_para_sp_pred
            # ####################################
            cur_sp_pred = supp_sent_prediction(predict_support_np_ith=predict_support_np_ith,
                                               example_dict=example_dict, batch_ids_ith=cur_id, thresholds=thresholds)
            ####################################
            # cur_sp_pred = supp_sent_prediction_hgn(predict_support_np_ith=predict_support_np_ith,
            #                                    example_dict=example_dict, batch_ids_ith=cur_id, thresholds=thresholds)
            # cur_sp_pred, cur_para_sp_pred = supp_sent_prediction_with_para_constraint(predict_para_support_np_ith=predict_para_support_np_ith,
            #                                                         predict_support_np_ith=predict_support_np_ith, example_dict=example_dict,
            #                                                         batch_ids_ith=cur_id, thresholds=thresholds)
            # total_para_sp_dict[cur_id] = cur_para_sp_pred
            ####################################
            # cur_sp_pred = [[] for _ in range(N_thresh)]
            # cur_id = batch['ids'][i]
            #
            # for j in range(predict_support_np.shape[1]):
            #     if j >= len(example_dict[cur_id].sent_names):
            #         break
            #
            #     for thresh_i in range(N_thresh):
            #         if predict_support_np[i, j] > thresholds[thresh_i]:
            #             cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])
            #             # print(example_dict[cur_id].sent_names[j])
            # ###################################

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []

                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        #####
        best_threshold_idx = -1
        #####
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                #####
                best_threshold_idx = thresh_i
                #####
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold, best_threshold_idx

    best_metrics, best_threshold, best_threshold_idx = choose_best_threshold(answer_dict, prediction_file)
    ##############++++++++++++
    doc_recall_metric = doc_recall_eval(doc_prediction=total_para_sp_dict, gold_file=dev_gold_file)
    total_inconsistent_number = supp_doc_sent_consistent_checker(predict_para_dict=total_para_sp_dict,
                                                               predicted_supp_sent_dict=total_sp_dict[best_threshold_idx], gold_file=dev_gold_file)
    ##############++++++++++++
    json.dump(best_metrics, open(eval_file, 'w'))

    return best_metrics, best_threshold, doc_recall_metric, total_inconsistent_number

########################################################################################################################
def compute_loss(args, batch, start, end, para, sent, ent, q_type):
    """
    y1: start position
    y2: end position
    q_type: question type
    is_support: whether the sentence is a support sentences: 1, yes, 0, no, -100, mask
    is_gold_para: whether the para is the support document: 1, yes, 0, no, -100, mask
    is_gold_ent: whether the entity is answer: -100, mask, other: ent index
    """
    # loss_labels = {'y1', 'y2', 'q_type', 'is_support', 'is_gold_ent', 'is_gold_para'}
    # loss_labels = {'is_gold_ent'}
    # for key, value in batch.items():
    #     if key in loss_labels:
    #         print('{}:{}'.format(key, value))
    #         print('prediction {}'.format(ent.shape))
    #         print('+' * 100)
    ans_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    loss_span = args.ans_lambda * (ans_criterion(start, batch['y1']) + ans_criterion(end, batch['y2']))
    loss_type = args.type_lambda * ans_criterion(q_type, batch['q_type'])
    loss_ent = args.ent_lambda * ans_criterion(ent, batch['is_gold_ent'].long())
    ####################################################################################################################
    sup_criterion = ATPLoss(reduction='mean')
    sent_mask = batch['sent_mask']
    batch_size = sent_mask.shape[0]
    sent_gold = batch['is_support']
    query_sent_pred = sent
    query_sent_gold = torch.cat([torch.zeros(batch_size, 1).to(sent_gold), sent_gold], dim=-1)
    query_sent_mask = torch.cat([torch.ones(batch_size, 1).to(sent_mask), sent_mask], dim=-1)
    print(sent_mask)
    print(sent_gold)

    loss_sup = args.sent_lambda * sup_criterion.forward(logits=query_sent_pred, labels=query_sent_gold, mask=query_sent_mask)

    para_mask = batch['para_mask']
    para_gold = batch['is_gold_para']
    print(para_mask)
    print(para_gold)

    query_para_pred = para
    query_para_gold = torch.cat([torch.zeros(batch_size, 1).to(para_gold), para_gold], dim=-1)
    query_para_mask = torch.cat([torch.ones(batch_size, 1).to(para_mask), para_mask], dim=-1)
    loss_para = args.para_lambda * sup_criterion.forward(logits=query_para_pred, labels=query_para_gold, mask=query_para_mask)
    ####################################################################################################################
    loss = loss_span + loss_type + loss_sup + loss_ent + loss_para

    return loss, loss_span, loss_type, loss_sup, loss_ent, loss_para