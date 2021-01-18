from transformers import LongformerModel, LongformerTokenizer
from torch import nn
from transformers import LongformerConfig
PRE_TAINED_LONFORMER_BASE = 'allenai/longformer-base-4096'
PRE_TAINED_LONFORMER_LARGE = 'allenai/longformer-large-4096'
FINE_TUNED_SQUADV_MODEL_NAME = 'valhalla/longformer-base-4096-finetuned-squadv1'
FINE_TUNED_SQUADV2_MODEL_NAME = 'mrm8488/longformer-base-4096-finetuned-squadv2'
FINE_TUNED_TRIVIQA_LARGE_MODEL_NAME = 'allenai/longformer-large-4096-finetuned-triviaqa'
FINE_TUNED_QA_MODEL_NAME = 'a-ware/longformer-QA'
########################################################################################################################
def get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE, do_lower_case=True):
    tokenizer = LongformerTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    return tokenizer
########################################################################################################################
class LongformerEncoder(LongformerModel):
    def __init__(self, config, project_dim: int = 0, seq_project=True):
        LongformerModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.seq_project = seq_project
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, attn_dropout: float = 0.1,
                     hidden_dropout: float = 0.1, seq_project=False, **kwargs) -> LongformerModel:
        cfg = LongformerConfig.from_pretrained(cfg_name if cfg_name else PRE_TAINED_LONFORMER_BASE)
        if attn_dropout != 0:
            cfg.attention_probs_dropout_prob = attn_dropout
        if hidden_dropout != 0:
            cfg.hidden_dropout_prob = hidden_dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, seq_project=seq_project, **kwargs)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            attention_mask=attention_mask,
                                                                            global_attention_mask=global_attention_mask)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             global_attention_mask=global_attention_mask)
        pooled_output = sequence_output[:, 0, :] ### get the first element [CLS], the second is the adding new token
        # print(pooled_output.shape, sequence_output.shape)
        if self.encode_proj:
            if self.seq_project:
                sequence_output = self.encode_proj(sequence_output)
                pooled_output = sequence_output[:, 0, :]
            else:
                pooled_output = self.encode_proj(pooled_output)
        # print(pooled_output.shape, sequence_output.shape)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size
########################################################################################################################