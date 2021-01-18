import pandas as pd
import logging
from argparse import Namespace
from pandas import DataFrame
from torch import Tensor as T
from datetime import date, datetime
import torch
from time import time
import swifter
MAX_ANSWER_DECODE_LEN = 30
MASK_VALUE = -1e9
########################################################################################################################
def loadJSONData(json_fileName)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(json_fileName, orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame
########################################################################################################################
def get_date_time():
    today = date.today()
    str_today = today.strftime('%b_%d_%Y')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    date_time_str = str_today + '_' + current_time
    return date_time_str
########################################################################################################################
def log_metrics(mode, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('{} {}: {:.4f}'.format(mode, metric, metrics[metric]))
########################################################################################################################
def sp_score(prediction, gold):
    cur_sp_pred = set(prediction)
    gold_sp_pred = set(gold)
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, prec, recall, f1
########################################################################################################################
def recall_computation(prediction, gold):
    gold_set = set(gold)
    gold_count = len(gold_set)
    tp = 0
    prediction_set = set(prediction)
    prediction = list(prediction_set)
    for pred in prediction:
        if pred in gold_set:
            tp = tp + 1
    recall = 1.0 * tp /gold_count
    em_recall = 1.0 if recall == 1.0 else 0.0
    return em_recall
########################################################################################################################
########################################################################################################################
def support_doc_test(doc_scores: T, doc_mask: T, top_k=2):
    batch_size, doc_num = doc_scores.shape[0], doc_scores.shape[1]
    assert top_k <= doc_num
    scores = torch.sigmoid(doc_scores)
    masked_doc_scores = scores.masked_fill(doc_mask == 0, -1)  ### mask
    argsort_doc = torch.argsort(masked_doc_scores, dim=1, descending=True)
    ####################################################################################################################
    pred_docs = []
    for idx in range(batch_size):
        pred_idxes_i = argsort_doc[idx].detach().tolist()
        pred_docs_i = pred_idxes_i[:top_k]
        pred_docs.append(pred_docs_i)
        # ==============================================================================================================
    doc_res = {'pred_doc': pred_docs}
    return doc_res
########################################################################################################################
def RetrievalEvaluation(data: DataFrame, args: Namespace, graph=True):
    golden_data = loadJSONData(json_fileName=args.raw_data)
    golden_data['e_id'] = range(0, golden_data.shape[0])
    merge_data = pd.concat([data.set_index('e_id'), golden_data.set_index('e_id')], axis=1, join='inner')
    def graph_process_row(row):
        predictions = row['doc_pred']
        ############
        top3_predictions = row['top3_doc']
        top4_predictions = row['top4_doc']
        top5_predictions = row['top5_doc']
        ############
        support_doc_titles = [x[0] for x in row['supporting_facts']]
        ctx_titles = [x[0] for x in row['context']]
        ############
        predicted_scores = row['doc_score']
        predicted_scores = predicted_scores[:(len(ctx_titles))]
        title_score_pair_list = list(zip(ctx_titles, predicted_scores))
        title_score_pair_list.sort(key=lambda x: x[1], reverse=True)
        ############
        pred_titles = [ctx_titles[_] for _ in predictions]
        em_recall = recall_computation(prediction=pred_titles, gold=support_doc_titles)
        ############
        top3_preds = [ctx_titles[_] for _ in top3_predictions if _ < len(ctx_titles)]
        top4_preds = [ctx_titles[_] for _ in top4_predictions if _ < len(ctx_titles)]
        top5_preds = [ctx_titles[_] for _ in top5_predictions if _ < len(ctx_titles)]
        ############
        top3_recall = recall_computation(prediction=top3_preds, gold=support_doc_titles)
        top4_recall = recall_computation(prediction=top4_preds, gold=support_doc_titles)
        top5_recall = recall_computation(prediction=top5_preds, gold=support_doc_titles)
        ############
        if graph:
            graph_type = row['graph_type']
            orig_type = row['type']
            combined_type = graph_type + '_' + orig_type
        else:
            combined_type = 'null'
        return em_recall, top3_recall, top4_recall, top5_recall, combined_type, title_score_pair_list
    res_names = ['em', 'top3', 'top4', 'top5', 'comb_type', 'ti_s_pair']
    merge_data[res_names] = merge_data.apply(lambda row: pd.Series(graph_process_row(row)), axis=1)
    recall_metric = merge_data[['em', 'top3', 'top4', 'top5']].mean()
    # temp_data = merge_data[merge_data['em']==1]
    temp_data = merge_data[merge_data['em'] == 1]
    values = temp_data['comb_type'].value_counts(dropna=False).keys().tolist()
    counts = temp_data['comb_type'].value_counts(dropna=False).tolist()
    value_dict = dict(zip(values, counts))
    ############
    doc_ids, para_score_pair = merge_data['_id'].tolist(), merge_data['ti_s_pair'].tolist()
    para_score_dict = dict(zip(doc_ids, para_score_pair))
    ############
    return recall_metric, value_dict, merge_data, para_score_dict
########################################################################################################################
########################################################################################################################
def graph_decoder_step(attn_scores: T, predictions: list):
    batch_size, query_doc_num = attn_scores.shape[0], attn_scores.shape[1]
    assert len(predictions) == batch_size
    graph_nodes_list = []
    graph_edge_list = []
    graph_type_list = []
    start_idx = 0
    for idx, preds in enumerate(predictions):
        attn_scores_i = attn_scores[idx]
        attn_mask_i = torch.zeros(query_doc_num, device=attn_scores.device)
        #############################
        preds = [x+1 for x in preds]
        #############################
        preds_i = [0] + preds
        attn_mask_i[preds_i] = 1
        graph_nodes, graph_edges, _ = graph_decoder(score_matrix=attn_scores_i, start_idx=start_idx, mask=attn_mask_i)
        assert len(graph_nodes) == 3 and len(graph_edges) == 2, 'nodes {} edges {}'.format(graph_nodes, graph_edges)
        # if graph_edges[1][0] == start_idx:
        if graph_edges[1][0] == 0:
            graph_type = 'branch'
        else:
            graph_type = 'chain'
        graph_nodes_list.append(graph_nodes)
        graph_edge_list.append(graph_edges)
        graph_type_list.append(graph_type)
    out_put = {'node': graph_nodes_list, 'edge': graph_edge_list, 'type': graph_type_list}
    return out_put
########################################################################################################################
def evaluation_graph_step(output_scores, batch, mode='val'):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    doc_attn_scores = output_scores['attn_score'] ### original score is used
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    doc_scores = output_scores['doc_score']
    doc_mask = batch['doc_lens']
    supp_doc_predictions = support_doc_test(doc_scores=doc_scores, doc_mask=doc_mask, top_k=2)
    top3_supp_doc_predictions = support_doc_test(doc_scores=doc_scores, doc_mask=doc_mask, top_k=3)
    top4_supp_doc_predictions = support_doc_test(doc_scores=doc_scores, doc_mask=doc_mask, top_k=4)
    top5_supp_doc_predictions = support_doc_test(doc_scores=doc_scores, doc_mask=doc_mask, top_k=5)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    batch_size = doc_scores.shape[0]
    batch_eids = batch['id'].squeeze().detach().tolist()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    assert len(supp_doc_predictions['pred_doc']) == batch_size
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    temp_scores = doc_attn_scores.clone()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    graph_out_put = graph_decoder_step(attn_scores=temp_scores, predictions=supp_doc_predictions['pred_doc'])
    out_put = {'attn_score': doc_attn_scores.detach().tolist(),
               'doc_score': doc_scores.detach().tolist(),
               'pred_docs': supp_doc_predictions['pred_doc'],
               'top3_docs': top3_supp_doc_predictions['pred_doc'],
               'top4_docs': top4_supp_doc_predictions['pred_doc'],
               'top5_docs': top5_supp_doc_predictions['pred_doc'],
               'graph_node': graph_out_put['node'],
               'graph_edge': graph_out_put['edge'],
               'graph_type': graph_out_put['type'],
               'ids': batch_eids}
    return out_put
########################################################################################################################
def graph_decoder(score_matrix: T, start_idx: int, score_mask:T=None, mask: T=None):
    ################################################################################
    graph_nodes = [start_idx]
    graph_edges = []
    ################################################################################
    sub_graph_scores = []
    scores = score_matrix.clone()
    if score_mask is not None:
        assert len(score_mask.shape) == 1
        scores = scores.masked_fill(score_mask == 0, MASK_VALUE)
        mask = mask.unsqueeze(dim=-1)
        scores = scores.masked_fill(score_mask == 0, MASK_VALUE)
    ################################################################################
    score_dim = scores.shape[-1]
    scores.fill_diagonal_(fill_value=MASK_VALUE)
    scores[:, start_idx] = MASK_VALUE
    ################################################################################
    if mask is not None:
        assert len(mask.shape) == 1
        graph_num = mask.sum().detach().item()
        scores = scores.masked_fill(mask == 0, MASK_VALUE)
        mask = mask.unsqueeze(dim=-1)
        scores = scores.masked_fill(mask == 0, MASK_VALUE)
    else:
        ################################################################################
        candidate_scores = scores[graph_nodes]
        max_idx = torch.argmax(candidate_scores)
        max_idx = max_idx.detach().item()
        row_idx, col_idx = max_idx // score_dim, max_idx % score_dim
        orig_row_idx = graph_nodes[row_idx]
        sub_graph_scores.append(score_matrix[orig_row_idx, col_idx])
        graph_edges.append((orig_row_idx, col_idx))
        ################################################################################
        graph_num = score_dim
    ################################################################################
    while True:
        candidate_scores = scores[graph_nodes]
        max_idx = torch.argmax(candidate_scores)
        max_idx = max_idx.detach().item()
        row_idx, col_idx = max_idx // score_dim, max_idx % score_dim
        if candidate_scores[row_idx, col_idx] == MASK_VALUE:
            break
        orig_row_idx = graph_nodes[row_idx]
        ################################################################################
        if mask is not None:
            graph_edges.append((orig_row_idx, col_idx))
            graph_nodes.append(col_idx)
            sub_graph_scores.append(score_matrix[orig_row_idx, col_idx])
            if len(graph_nodes) == graph_num:
                break
        else:
            if score_matrix[orig_row_idx, col_idx] < 0 or len(graph_nodes) >= graph_num:
                break
            else:
                graph_edges.append((orig_row_idx, col_idx))
                graph_nodes.append(col_idx)
                sub_graph_scores.append(score_matrix[orig_row_idx, col_idx])
        ################################################################################
        scores[:, col_idx] = MASK_VALUE
        ################################################################################
    if len(sub_graph_scores)==0:
        print(score_matrix, mask)
    sub_graph_score = torch.stack(sub_graph_scores).sum().detach().item()
    return graph_nodes, graph_edges, sub_graph_score