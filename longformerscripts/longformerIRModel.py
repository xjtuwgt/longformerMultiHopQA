from torch import Tensor as T
from argparse import Namespace
from torch import nn
import math
import torch
import pytorch_lightning as pl
import logging
import torch.nn.functional as F
import copy
from longformerscripts.longformerUtils import LongformerEncoder, get_hotpotqa_longformer_tokenizer
########################################################################################################################
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
MASK_VALUE = -1e9
########################################################################################################################
class GraphPPRScoreFunc(nn.Module):
    def __init__(self, d_model, alpha=0.25, hop_num=6, drop_out=0.1, smooth_factor=1e-7):
        super(GraphPPRScoreFunc, self).__init__()
        self.d_model = d_model
        self.linears = clones(nn.Linear(self.d_model, self.d_model), 2)
        self.alpha = alpha
        self.hop_num = hop_num
        self.ppr = hop_num > 1
        self.smooth_factor=smooth_factor
        self.drop_out = nn.Dropout(p=drop_out)
        self.init()

    def init(self):
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight.data)

    def forward(self, x: T, x_mask: T=None):
        if x_mask is not None:
            x_mask = x_mask.unsqueeze(dim=1)
        batch_size = x.shape[0]
        query_doc_num = x.shape[1]
        query, key = [l(y).view(batch_size, -1, self.d_model)
                                                      for l, y in zip(self.linears, (x, x))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(self.d_model)
        if x_mask is not None:
            scores = scores.masked_fill(x_mask==0, MASK_VALUE)
        #########################################
        eye_mask = torch.eye(query_doc_num, query_doc_num, device=x.device).byte()
        scores = scores.masked_fill(eye_mask==1, MASK_VALUE)
        #########################################
        p_attn = F.softmax(scores, dim=-1)
        if self.ppr:
            ppr_attn = self.ppr_propagation(p_attn=p_attn)
        else:
            ppr_attn = p_attn
        doc_score = ppr_attn[:, 0, :][:, torch.arange(1, query_doc_num)]
        doc_score = F.normalize(doc_score, p=1, dim=1)  ## normalize to sum = 1
        doc_score = torch.clamp(doc_score, self.smooth_factor, 1.0 - self.smooth_factor)
        return doc_score, scores

    def ppr_propagation(self, p_attn: T):
        query_doc_num = p_attn.shape[-1]
        temp_attn = self.alpha * torch.eye(query_doc_num, device=p_attn.device)
        ppr_attn_matrix = temp_attn
        # for r in range(1, self.hop_num):
        for r in range(0, self.hop_num):
            temp_attn = torch.matmul(self.drop_out(temp_attn), (1 - self.alpha) * p_attn)
            ppr_attn_matrix = ppr_attn_matrix + temp_attn
        return ppr_attn_matrix
########################################################################################################################
class LongformerGraphRetrievalModel(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.hparams = args
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.tokenizer = get_hotpotqa_longformer_tokenizer(model_name=self.hparams.pretrained_cfg_name)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        longEncoder = LongformerEncoder.init_encoder(cfg_name=self.hparams.pretrained_cfg_name, projection_dim=self.hparams.project_dim,
                                                     hidden_dropout=self.hparams.input_drop, attn_dropout=self.hparams.attn_drop,
                                                     seq_project=self.hparams.seq_project)
        longEncoder.resize_token_embeddings(len(self.tokenizer))
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.hparams.frozen_layer_num > 0:
            modules = [longEncoder.embeddings, *longEncoder.encoder.layer[:self.hparams.frozen_layer_num]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            logging.info('Frozen the first {} layers'.format(self.hparams.frozen_layer_num))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if 'base' in self.hparams.pretrained_cfg_name:
            self.fix_encoder = self.hparams.frozen_layer_num == 12
        elif 'large' in self.hparams.pretrained_cfg_name:
            self.fix_encoder = self.hparams.frozen_layer_num == 24
        else:
            raise ValueError('Pre-trained model %s not supported' % self.hparams.pretrained_cfg_name)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.longformer = longEncoder #### LongFormer encoder
        self.hidden_size = longEncoder.get_out_size()
        self.doc_graph_score = GraphPPRScoreFunc(d_model=self.hidden_size, alpha=self.hparams.ppr_alpha, hop_num=self.hparams.hop_num)
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.hparams = args
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.loss_type = self.hparams.loss_type
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.mask_value = MASK_VALUE
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def get_representation(sub_model: LongformerEncoder, ids: T, attn_mask: T, global_attn_mask: T,
                           fix_encoder: bool = False) -> (T, T, T):
        sequence_output = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, _, _ = sub_model.forward(input_ids=ids, attention_mask=attn_mask,
                                                              global_attention_mask=global_attn_mask)
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, _, _ = sub_model.forward(input_ids=ids, attention_mask=attn_mask,
                                                          global_attention_mask=global_attn_mask)
        return sequence_output
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def forward(self, sample):
        ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask = sample['ctx_encode'], sample['ctx_attn_mask'], sample['ctx_global_mask']
        sequence_output = self.get_representation(self.longformer, ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask, self.fix_encoder)
        return sequence_output
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def supp_doc_prediction(self, doc_embed: T, doc_mask: T):
        doc_score, attn_scores = self.doc_graph_score.forward(x=doc_embed, x_mask=doc_mask)
        return doc_score, attn_scores
    ####################################################################################################################
    def score_computation(self, sample):
        sequence_output = self.forward(sample=sample)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_embed = sequence_output[:, 1, :].unsqueeze(dim=1)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_start_positions, doc_end_positions = sample['doc_start'], sample['doc_end']
        batch_size, doc_num = doc_start_positions.shape
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_batch_idx = torch.arange(0, batch_size, device=sequence_output.device).view(batch_size, 1).repeat(1, doc_num)
        doc_start_embed = sequence_output[doc_batch_idx, doc_start_positions]
        doc_end_embed = sequence_output[doc_batch_idx, doc_end_positions]
        doc_embed = (doc_start_embed + doc_end_embed)/2.0
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_doc_embed = torch.cat([query_embed, doc_embed], dim=1)
        doc_lens = sample['doc_lens']
        doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
        query_doc_mask = F.pad(doc_mask, (1, 0, 0, 0), 'constant', 1)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_scores, attn_scores = self.supp_doc_prediction(doc_embed=query_doc_embed, doc_mask=query_doc_mask)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output_score = {'doc_score': doc_scores, 'attn_score': attn_scores}
        return output_score
