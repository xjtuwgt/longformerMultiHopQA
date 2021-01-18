from __future__ import absolute_import, division, print_function
from hgntransformers import (BertConfig, BertTokenizer, BertModel,
                             RobertaConfig, RobertaTokenizer, RobertaModel,
                             AlbertConfig, AlbertTokenizer, AlbertModel)
from hgntransformers import (BertModel, XLNetModel, RobertaModel)
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig
from transformers import configuration_longformer

############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())
ALL_LONG_MODELS = sum((tuple(configuration_longformer.LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
    'longformer': (LongformerConfig, LongformerModel, LongformerTokenizer)
}
