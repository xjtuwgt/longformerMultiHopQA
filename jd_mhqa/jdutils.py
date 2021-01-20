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
from csr_mhqa.utils import convert_to_tokens
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
def supp_doc_prediction(predict_para_support_np_ith, example_dict, batch_ids_ith):
    arg_order_ids = np.argsort(predict_para_support_np_ith)[::-1].tolist()
    cand_para_names = example_dict[batch_ids_ith].para_names
    assert len(cand_para_names) >=2
    cur_sp_para_pred = [cand_para_names[arg_order_ids[0]], cand_para_names[arg_order_ids[1]]]
    return cur_sp_para_pred
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_doc_sent_consistent_checker(predict_para_dict: dict, predicted_supp_sent_dict: dict):
    def consistent_checker(predict_para_support, pred_titles):
        if len(pred_titles) != len(predict_para_support):
            return False
        for p_title in pred_titles:
            if p_title not in predict_para_support:
                return False
        return True
    total_inconsist_num = 0
    for para_id, predict_para in predict_para_dict.items():
        predict_supp_sents = predicted_supp_sent_dict[para_id]
        pred_titles = list(set([x[0] for x in predict_supp_sents]))
        whether_consist = consistent_checker(predict_para_support=predict_para, pred_titles=pred_titles)
        if not whether_consist:
            total_inconsist_num = total_inconsist_num + 1
    return total_inconsist_num
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_sent_prediction_with_constraint(predict_support_np_ith, example_dict, batch_ids_ith, thresholds):
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
            ####################################
            predict_para_support_np_ith = predict_para_support_np[i]
            cur_para_sp_pred = supp_doc_prediction(predict_para_support_np_ith=predict_para_support_np_ith, example_dict=example_dict, batch_ids_ith=cur_id)
            total_para_sp_dict[cur_id] = cur_para_sp_pred
            ####################################
            predict_support_np_ith = predict_support_np[i]
            cur_sp_pred = supp_sent_prediction(predict_support_np_ith=predict_support_np_ith,
                                               example_dict=example_dict, batch_ids_ith=cur_id, thresholds=thresholds)
            ####################################
            # ###################################
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
                                                               predicted_supp_sent_dict=total_sp_dict[best_threshold_idx])
    ##############++++++++++++
    json.dump(best_metrics, open(eval_file, 'w'))

    return best_metrics, best_threshold, doc_recall_metric, total_inconsistent_number

