import envs

from transformers import (BertConfig, BertTokenizer, BertModel,
                          RobertaConfig, RobertaTokenizer, RobertaModel,
                          AlbertConfig, AlbertTokenizer, AlbertModel)
from transformers import (BertModel, XLNetModel, RobertaModel)

from transformers import (configuration_bert, configuration_roberta, configuration_albert)

############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf) for conf in (configuration_bert.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys(),
                                           configuration_roberta.ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys(),
                                           configuration_albert.ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
}
