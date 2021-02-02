from models.layers import mean_pooling, BiAttention, LSTMWrapper, GraphBlock, GatedAttention, PredictionLayer
from torch import nn
from csr_mhqa.utils import count_parameters


class HierarchicalGraphNetwork(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        self.hidden_dim = config.hidden_dim

        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=1,
                                     dropout=config.lstm_drop)

        self.graph_blocks = nn.ModuleList()
        for _ in range(self.config.num_gnn_layers):
            self.graph_blocks.append(GraphBlock(self.config.q_attn, config))

        self.ctx_attention = GatedAttention(input_dim=config.hidden_dim*2,
                                            memory_dim=config.hidden_dim if config.q_update else config.hidden_dim*2,
                                            hid_dim=self.config.ctx_attn_hidden_dim,
                                            dropout=config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn)

        q_dim = self.hidden_dim if config.q_update else config.input_dim

        self.predict_layer = PredictionLayer(self.config, q_dim)

    def forward(self, batch, return_yp):
        query_mapping = batch['query_mapping']
        context_encoding = batch['context_encoding']
        print(query_mapping.shape)
        print(query_mapping)

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        print(trunc_query_mapping.shape)
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        print(trunc_query_state.shape)
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)
        print(query_vec.shape)

        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)
        print('attn_output {}'.format(attn_output.shape))
        print('trunc_query_state {}'.format(trunc_query_state.shape))

        input_state = self.bi_attn_linear(attn_output) # N x L x d
        print('attn_output {}'.format(attn_output.shape))
        print('input_state {}'.format(input_state.shape))
        input_state = self.sent_lstm(input_state, batch['context_lens'])
        print('input_state {}'.format(input_state.shape))
        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        print('query vec {}'.format(query_vec.shape))
        para_logits, sent_logits = [], []
        para_predictions, sent_predictions, ent_predictions = [], [], []

        for l in range(self.config.num_gnn_layers):
            new_input_state, graph_state, graph_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit = self.graph_blocks[l](batch, input_state, query_vec)

            print('input_state {} new input_state {} query vec {}'.format(input_state.shape, new_input_state.shape, query_vec.shape))
            print('input {}'.format(input_state))
            print('new input {}'.format(new_input_state))


            para_logits.append(para_logit)
            sent_logits.append(sent_logit)
            para_predictions.append(para_prediction)
            sent_predictions.append(sent_prediction)
            ent_predictions.append(ent_logit)

        input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))
        predictions = self.predict_layer(batch, input_state, sent_logits[-1], packing_mask=query_mapping, return_yp=return_yp)

        if return_yp:
            start, end, q_type, yp1, yp2 = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1], yp1, yp2
        else:
            start, end, q_type = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1]
