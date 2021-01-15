from __future__ import absolute_import, division, print_function
from longformerscripts.longformerIRModel import LongformerGraphRetrievalModel
import argparse
from longformerDataUtils.fullHotpotQADataSet import HotpotTestDataset
from longformerscripts.longformerUtils import get_hotpotqa_longformer_tokenizer
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from longformerDataUtils.ioutils import loadJSONData
from longformerscripts.longformerIREvaluation import evaluation_graph_step, get_date_time, RetrievalEvaluation
from pandas import DataFrame
from time import time

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluating Longformer based retrieval Model')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--raw_data', default=None,type=str, required=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--input_data', default=None, type=str, required=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--eval_ckpt", default=None, type=str, required=True,
                        help="evaluation checkpoint")
    parser.add_argument("--model_type", default='Longformer', type=str, help="longformer retrieval model")
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--max_doc_num', default=10, type=int)
    parser.add_argument('--test_log_steps', default=10, type=int)
    parser.add_argument('--cpu_num', default=24, type=int)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return parser.parse_args(args)

def batch2device(batch, device):
    sample = dict()
    for key, value in batch.items():
        sample[key] = value.to(device)
    return sample

def rank_paras(data, pred_score):
    ranked_paras = dict()
    cur_ptr = 0
    for case in tqdm(data):
        key = case['_id']
        tem_ptr = cur_ptr

        all_paras = []
        while cur_ptr < tem_ptr + len(case['context']):
            score = pred_score.loc[cur_ptr, 'prob'].item()
            all_paras.append((case['context'][cur_ptr - tem_ptr][0], score))
            cur_ptr += 1

        sorted_all_paras = sorted(all_paras, key=lambda x: x[1], reverse=True)
        ranked_paras[key] = sorted_all_paras

    return ranked_paras

def test_data_loader(args):
    tokenizer = get_hotpotqa_longformer_tokenizer()
    test_data_frame = loadJSONData(PATH=args.data_path, json_fileName=args.test_data_name)
    test_data_frame['e_id'] = range(0, test_data_frame.shape[0]) ## for alignment
    test_data = HotpotTestDataset(data_frame=test_data_frame, tokenizer=tokenizer, max_doc_num=10)
    dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotTestDataset.collate_fn
    )
    return dataloader
########################################################################################################################
def graph_retrieval_test_procedure(model, test_data_loader, args, device):
    model.freeze()
    out_puts = []
    start_time = time()
    total_steps = len(test_data_loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            batch = batch2device(batch=batch, device=device)
            output_scores = model.score_computation(sample=batch)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            output = evaluation_graph_step(output_scores=output_scores, batch=batch, mode='test')
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if (batch_idx + 1) % args.test_log_steps == 0:
                print('Evaluating the model... {}/{} in {:.4f} seconds'.format(batch_idx + 1, total_steps, time()-start_time))
            out_puts.append(output)
    example_ids = []
    doc_scores = []
    pred_docs = []
    top3_docs = []
    top4_docs = []
    top5_docs = []
    graph_nodes = []
    graph_edges = []
    graph_types = []
    graph_attn_scores = []
    for output in out_puts:
        example_ids += output['ids']
        doc_scores += output['doc_score']
        pred_docs += output['pred_docs']
        graph_attn_scores += output['attn_score']
        graph_nodes += output['graph_node']
        graph_edges += output['graph_edge']
        graph_types += output['graph_type']
        top3_docs += output['top3_docs']
        top4_docs += output['top4_docs']
        top5_docs += output['top5_docs']
    # print(len(example_ids), len(doc_scores), len(pred_docs), len(graph_attn_scores), len(graph_nodes), len(graph_edges), len(graph_types))
    result_dict = {'e_id': example_ids,##for alignment
                   'doc_pred': pred_docs,
                   'top3_doc': top3_docs,
                   'top4_doc': top4_docs,
                   'top5_doc': top5_docs,
                   'graph_node': graph_nodes,
                   'graph_edge': graph_edges,
                   'graph_type': graph_types,
                   'graph_attn': graph_attn_scores,
                   'doc_score': doc_scores}  ## for detailed results checking
    res_data_frame = DataFrame(result_dict)
    return res_data_frame
########################################################################################################################
def main(args):
    if args.gpus > 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    hotpotIR_model = LongformerGraphRetrievalModel.load_from_checkpoint(checkpoint_path=args.eval_ckpt)
    hotpotIR_model = hotpotIR_model.to(device)
    print('Model Parameter Configuration:')
    for name, param in hotpotIR_model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)
    print("Model hype-parameter information...")
    for key, value in vars(args).items():
        print('Hype-parameter\t{} = {}'.format(key, value))
    print('*' * 75)
    test_data = test_data_loader(args=args)
    res_df = graph_retrieval_test_procedure(model=hotpotIR_model, test_data_loader=test_data, args=args, device=device)
    ####################################################################################################################
    metric_name = get_date_time() + '_doc_gir'
    metric, comb_dict, res_df = RetrievalEvaluation(data=res_df, args=args, graph=True)
    print('Doc retrieval metrics = {}'.format(metric))
    for key, value in comb_dict.items():
        print('{}:{}'.format(key, value))
    print('*'*75)
    ####################################################################################################################
    save_result_name = os.path.join(args.log_path, args.log_name, metric_name + '.json')
    res_df.to_json(save_result_name)
    ####################################################################################################################
    print('Saving {} records into {}'.format(res_df.shape, save_result_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)
