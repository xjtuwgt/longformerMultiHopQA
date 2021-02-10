import torch
from torch import nn
from torch.autograd import Variable
from models.layers import OutputLayer
from csr_mhqa.utils import get_weights, get_act
import torch.nn.functional as F
import numpy as np

def encoder_graph_node_feature(batch, input_state, hidden_dim):
    sent_start_mapping = batch['sent_start_mapping']
    sent_end_mapping = batch['sent_end_mapping']
    para_start_mapping = batch['para_start_mapping']
    para_end_mapping = batch['para_end_mapping']
    ent_start_mapping = batch['ent_start_mapping']
    ent_end_mapping = batch['ent_end_mapping']

    para_start_output = torch.bmm(para_start_mapping, input_state[:, :, hidden_dim:])  # N x max_para x d
    para_end_output = torch.bmm(para_end_mapping, input_state[:, :, :hidden_dim])  # N x max_para x d
    para_state = torch.cat([para_start_output, para_end_output], dim=-1)  # N x max_para x 2d

    sent_start_output = torch.bmm(sent_start_mapping, input_state[:, :, hidden_dim:])  # N x max_sent x d
    sent_end_output = torch.bmm(sent_end_mapping, input_state[:, :, :hidden_dim])  # N x max_sent x d
    sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)  # N x max_sent x 2d

    ent_start_output = torch.bmm(ent_start_mapping, input_state[:, :, hidden_dim:])  # N x max_ent x d
    ent_end_output = torch.bmm(ent_end_mapping, input_state[:, :, :hidden_dim])  # N x max_ent x d
    ent_state = torch.cat([ent_start_output, ent_end_output], dim=-1)  # N x max_ent x 2d

    graph_state_dict = {'para_state': para_state, 'sent_state': sent_state, 'ent_state': ent_state}
    return graph_state_dict


class GraphBlock(nn.Module):
    def __init__(self, q_attn, config, input_dim, hidden_dim):
        super(GraphBlock, self).__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.gat = AttentionLayer(in_dim=self.input_dim,  hid_dim=self.hidden_dim, n_head=config.num_gnn_heads,
                                  q_attn=q_attn, config=self.config)

    def forward(self, batch, query_vec, graph_state_dict):
        para_state = graph_state_dict['para_state']
        sent_state = graph_state_dict['sent_state']
        ent_state = graph_state_dict['ent_state']

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()
        graph_state = torch.cat([query_vec.unsqueeze(1), para_state, sent_state, ent_state], dim=1)
        node_mask = torch.cat([torch.ones(N, 1).to(self.config.device), batch['para_mask'], batch['sent_mask'], batch['ent_mask']], dim=-1).unsqueeze(-1)
        graph_adj = batch['graphs']
        assert graph_adj.size(1) == node_mask.size(1)
        graph_state = self.gat(graph_state, graph_adj, node_mask=node_mask, query_vec=query_vec) # N x (1+max_para+max_sent) x d
        ent_state = graph_state[:, 1+max_para_num+max_sent_num:, :]
        ##############################################
        para_state = graph_state[:, 1:1+max_para_num, :]
        sent_state = graph_state[:, 1+max_para_num:1+max_para_num+max_sent_num, :]
        ##############################################
        query_vec = graph_state[:, 0, :].squeeze(1)
        ##############################################
        graph_state_dict['para_state'] = para_state
        graph_state_dict['sent_state'] = sent_state
        graph_state_dict['ent_state'] = ent_state
        ##############################################
        return graph_state, graph_state_dict, node_mask, query_vec

class GATSelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, config, q_attn=False, head_id=0):
        """ One head GAT """
        super(GATSelfAttention, self).__init__()
        self.config = config
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = self.config.gnn_drop
        self.q_attn = q_attn
        self.query_dim = in_dim
        self.n_type = self.config.num_edge_type

        self.head_id = head_id
        self.step = 0

        self.W_type = nn.ParameterList()
        self.a_type = nn.ParameterList()
        self.qattn_W1 = nn.ParameterList()
        self.qattn_W2 = nn.ParameterList()
        for i in range(self.n_type):
            self.W_type.append(get_weights((in_dim, out_dim)))
            self.a_type.append(get_weights((out_dim * 2, 1)))
            if self.q_attn:
                self.qattn_W1.append(get_weights((in_dim, out_dim * 2)))
                self.qattn_W2.append(get_weights((out_dim * 2, out_dim * 2)))

        self.act = get_act('lrelu:0.2')

    def forward(self, input_state, adj, node_mask=None, query_vec=None):
        zero_vec = torch.zeros_like(adj)
        scores = torch.zeros_like(adj)

        for i in range(self.n_type):
            h = torch.matmul(input_state, self.W_type[i])
            h = F.dropout(h, self.dropout, self.training)
            N, E, d = h.shape

            a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim=-1)
            a_input = a_input.view(-1, E, E, 2*d)

            # print(a_input.shape)

            if self.q_attn:
                q_gate = F.relu(torch.matmul(query_vec, self.qattn_W1[i]))
                q_gate = torch.sigmoid(torch.matmul(q_gate, self.qattn_W2[i]))
                a_input = a_input * q_gate[:, None, None, :]
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            else:
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            # print(i, input_state.shape, scores.shape, adj.shape, score.shape)
            scores += torch.where(adj == i+1, score, zero_vec.to(score.dtype))

        zero_vec = -1e30 * torch.ones_like(scores)
        scores = torch.where(adj > 0, scores, zero_vec.to(scores.dtype))

        # Ahead Alloc
        if node_mask is not None:
            h = h * node_mask

        coefs = F.softmax(scores, dim=2)  # N * E * E
        h = coefs.unsqueeze(3) * h.unsqueeze(2)  # N * E * E * d
        h = torch.sum(h, dim=1)
        return h


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_head, q_attn, config):
        super(AttentionLayer, self).__init__()
        assert hid_dim % n_head == 0
        self.dropout = config.gnn_drop

        self.attn_funcs = nn.ModuleList()
        for i in range(n_head):
            self.attn_funcs.append(
                GATSelfAttention(in_dim=in_dim, out_dim=hid_dim // n_head, config=config, q_attn=q_attn, head_id=i))

        if in_dim != hid_dim:
            self.align_dim = nn.Linear(in_dim, hid_dim)
            nn.init.xavier_uniform_(self.align_dim.weight, gain=1.414)
        else:
            self.align_dim = lambda x: x

    def forward(self, input, adj, node_mask=None, query_vec=None):
        hidden_list = []
        for attn in self.attn_funcs:
            h = attn(input, adj, node_mask=node_mask, query_vec=query_vec)
            hidden_list.append(h)

        h = torch.cat(hidden_list, dim=-1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        return h

class ParaSentEntPredictionLayer(nn.Module):
    def __init__(self, config, hidden_dim):
        super(ParaSentEntPredictionLayer, self).__init__()
        self.hidden_dim = hidden_dim

        self.para_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        self.entity_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)

        self.para_query_pooler = LinearAttentionPooLayer(hidden_dim=self.hidden_dim)
        self.sent_query_pooler = LinearAttentionPooLayer(hidden_dim=self.hidden_dim)


    def forward(self, batch, graph_state_dict, query_vec):

        para_state = graph_state_dict['para_state']
        sent_state = graph_state_dict['sent_state']
        ent_state = graph_state_dict['ent_state']

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()

        query_sent_vec = self.sent_query_pooler.forward(query=query_vec, key=sent_state, value=sent_state, mask=batch['sent_mask'])
        query_para_vec = self.para_query_pooler.forward(query=query_vec, key=para_state, value=para_state, mask=batch['para_mask'])

        query_sent_state = torch.cat([query_sent_vec.unsqueeze(1), sent_state], dim=1)
        query_para_state = torch.cat([query_para_vec.unsqueeze(1), para_state], dim=1)

        # query_sent_state = torch.cat([query_vec.unsqueeze(1), sent_state], dim=1)
        # query_para_state = torch.cat([query_vec.unsqueeze(1), para_state], dim=1)

        ent_logit = self.entity_mlp(ent_state).view(N, -1)
        ent_logit = ent_logit - 1e30 * (1 - batch['ans_cand_mask'])

        sent_logit = self.sent_mlp(query_sent_state).squeeze(dim=-1)
        para_logit = self.para_mlp(query_para_state).squeeze(dim=-1)
        return para_logit, sent_logit, ent_logit

class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.config = config
        input_dim = config.ctx_attn_hidden_dim
        h_dim = config.hidden_dim

        self.hidden = h_dim

        self.start_linear = OutputLayer(input_dim, config, num_answer=1)
        self.end_linear = OutputLayer(input_dim, config, num_answer=1)
        self.type_linear = OutputLayer(input_dim, config, num_answer=4)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, sent_logits, packing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        sent_mapping = batch['sent_mapping']
        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        type_prediction = self.type_linear(context_input[:, 0, :])##cls based prediction

        if not return_yp:
            return start_prediction, end_prediction, type_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, type_prediction, yp1, yp2

class LinearAttentionPooLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(LinearAttentionPooLayer, self).__init__()
        self.query_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, 1),
        )

        self.key_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, 1),
        )
        self.value_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        query_score = self.key_score(query)
        key_score = self.key_score(key).squeeze(-1)
        # print(query_score.shape)
        # print(key_score.shape)
        attn_score = key_score + query_score
        if mask is not None:
            attn_score = attn_score - 1e30 * (1 - mask)
        attn_prob = F.softmax(attn_score, dim=-1).unsqueeze(dim=1)
        # print(attn_prob.shape)
        # print(attn_prob)
        value_map = self.value_linear(value)
        res = torch.matmul(attn_prob, value_map).squeeze(dim=1)
        return res