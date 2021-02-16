from __future__ import absolute_import, division, print_function
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from hgntransformers import (BertConfig, BertTokenizer, BertModel,
                             RobertaConfig, RobertaTokenizer, RobertaModel,
                             AlbertConfig, AlbertTokenizer, AlbertModel)
from hgntransformers import (BertModel, XLNetModel, RobertaModel)

############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
    'longformer': (LongformerConfig, LongformerModel, LongformerTokenizer),
    'unifiedqa': (T5Config, T5ForConditionalGeneration, AutoTokenizer)
}
