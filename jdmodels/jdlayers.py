import torch
from torch import nn
from torch.autograd import Variable
from models.layers import AttentionLayer, OutputLayer

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


# class GraphBlock(nn.Module):
#     def __init__(self, q_attn, config):
#         super(GraphBlock, self).__init__()
#         self.config = config
#         self.hidden_dim = config.hidden_dim
#
#         if self.config.q_update:
#             self.gat_linear = nn.Linear(self.hidden_dim*2, self.hidden_dim)
#         else:
#             self.gat_linear = nn.Linear(self.config.input_dim, self.hidden_dim*2)
#
#         if self.config.q_update:
#             self.gat = AttentionLayer(self.hidden_dim, self.hidden_dim, config.num_gnn_heads, q_attn=q_attn, config=self.config)
#             self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
#             self.entity_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
#         else:
#             self.gat = AttentionLayer(self.hidden_dim*2, self.hidden_dim*2, config.num_gnn_heads, q_attn=q_attn, config=self.config)
#             self.sent_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)
#             self.entity_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)
#
#     def forward(self, batch, query_vec, graph_state_dict=None):
#         para_state = graph_state_dict['para_state']
#         sent_state = graph_state_dict['sent_state']
#         ent_state = graph_state_dict['ent_state']
#
#         N, max_para_num, _ = para_state.size()
#         _, max_sent_num, _ = sent_state.size()
#         _, max_ent_num, _ = ent_state.size()
#
#         if self.config.q_update:
#             graph_state = self.gat_linear(torch.cat([para_state, sent_state, ent_state], dim=1)) # N * (max_para + max_sent + max_ent) * d
#             graph_state = torch.cat([query_vec.unsqueeze(1), graph_state], dim=1)
#         else:
#             graph_state = self.gat_linear(query_vec)
#             graph_state = torch.cat([graph_state.unsqueeze(1), para_state, sent_state, ent_state], dim=1)
#         node_mask = torch.cat([torch.ones(N, 1).to(self.config.device), batch['para_mask'], batch['sent_mask'], batch['ent_mask']], dim=-1).unsqueeze(-1)
#
#         graph_adj = batch['graphs']
#         assert graph_adj.size(1) == node_mask.size(1)
#
#         graph_state = self.gat(graph_state, graph_adj, node_mask=node_mask, query_vec=query_vec) # N x (1+max_para+max_sent) x d
#         ent_state = graph_state[:, 1+max_para_num+max_sent_num:, :]
#
#         gat_logit = self.sent_mlp(graph_state[:, :1+max_para_num+max_sent_num, :]) # N x max_sent x 1
#         para_logit = gat_logit[:, 1:1+max_para_num, :].contiguous() ## para logit computation and sentence logit prediction share the same mlp
#         sent_logit = gat_logit[:, 1+max_para_num:, :].contiguous() ## para logit computation and sentence logit prediction share the same mlp
#
#         query_vec = graph_state[:, 0, :].squeeze(1)
#
#         ent_logit = self.entity_mlp(ent_state).view(N, -1)
#         ent_logit = ent_logit - 1e30 * (1 - batch['ans_cand_mask'])
#
#         para_logits_aux = Variable(para_logit.data.new(para_logit.size(0), para_logit.size(1), 1).zero_())
#         para_prediction = torch.cat([para_logits_aux, para_logit], dim=-1).contiguous()
#
#         sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
#         sent_prediction = torch.cat([sent_logits_aux, sent_logit], dim=-1).contiguous()
#
#         ##############################################
#         graph_state_dict['para_state'] = para_state
#         graph_state_dict['sent_state'] = sent_state
#         graph_state_dict['ent_state'] = ent_state
#         ##############################################
#         return graph_state, graph_state_dict, node_mask, sent_state, query_vec, para_logit, para_prediction, \
#             sent_logit, sent_prediction, ent_logit


class GraphBlock(nn.Module):
    def __init__(self, q_attn, config, q_input_dim):
        super(GraphBlock, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        if self.config.q_update:
            self.gat_linear = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        else:
            self.gat_linear = nn.Linear(q_input_dim, self.hidden_dim*2)

        if self.config.q_update:
            self.gat = AttentionLayer(self.hidden_dim, self.hidden_dim, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        else:
            self.gat = AttentionLayer(self.hidden_dim*2, self.hidden_dim*2, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            self.sent_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)

    def forward(self, batch, query_vec, graph_state_dict=None):
        para_state = graph_state_dict['para_state']
        sent_state = graph_state_dict['sent_state']
        ent_state = graph_state_dict['ent_state']

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()

        if self.config.q_update:
            graph_state = self.gat_linear(torch.cat([para_state, sent_state, ent_state], dim=1)) # N * (max_para + max_sent + max_ent) * d
            graph_state = torch.cat([query_vec.unsqueeze(1), graph_state], dim=1)
        else:
            graph_state = self.gat_linear(query_vec)
            graph_state = torch.cat([graph_state.unsqueeze(1), para_state, sent_state, ent_state], dim=1)
        node_mask = torch.cat([torch.ones(N, 1).to(self.config.device), batch['para_mask'], batch['sent_mask'], batch['ent_mask']], dim=-1).unsqueeze(-1)

        graph_adj = batch['graphs']
        assert graph_adj.size(1) == node_mask.size(1)

        graph_state = self.gat(graph_state, graph_adj, node_mask=node_mask, query_vec=query_vec) # N x (1+max_para+max_sent) x d
        ent_state = graph_state[:, 1+max_para_num+max_sent_num:, :]

        gat_logit = self.sent_mlp(graph_state[:, :1+max_para_num+max_sent_num, :]) # N x max_sent x 1
        para_logit = gat_logit[:, 1:1+max_para_num, :].contiguous() ## para logit computation and sentence logit prediction share the same mlp
        sent_logit = gat_logit[:, 1+max_para_num:, :].contiguous() ## para logit computation and sentence logit prediction share the same mlp

        query_vec = graph_state[:, 0, :].squeeze(1)

        ent_logit = self.entity_mlp(ent_state).view(N, -1)
        ent_logit = ent_logit - 1e30 * (1 - batch['ans_cand_mask'])

        para_logits_aux = Variable(para_logit.data.new(para_logit.size(0), para_logit.size(1), 1).zero_())
        para_prediction = torch.cat([para_logits_aux, para_logit], dim=-1).contiguous()

        sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
        sent_prediction = torch.cat([sent_logits_aux, sent_logit], dim=-1).contiguous()

        ##############################################
        graph_state_dict['para_state'] = para_state
        graph_state_dict['sent_state'] = sent_state
        graph_state_dict['ent_state'] = ent_state
        ##############################################
        return graph_state, graph_state_dict, node_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit