import envs

from hgntransformers import (BertConfig, BertTokenizer, BertModel,
                             RobertaConfig, RobertaTokenizer, RobertaModel,
                             AlbertConfig, AlbertTokenizer, AlbertModel)
from hgntransformers import (BertModel, XLNetModel, RobertaModel)
from transformers import LongformerTokenizer, LongformerConfig, configuration_longformer
from transformers import LongformerModel

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
