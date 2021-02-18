import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import logging
import torch
import sys
from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from lightningmodel.lightningHGN import lightningHGN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def set_args(cmd_argv):
    parser = default_train_parser()
    # args_config_provided = parser.parse_args(sys.argv[1:])
    args_config_provided = parser.parse_args(cmd_argv)
    if args_config_provided.config_file is not None:
        # argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
        argv = json_to_argv(args_config_provided.config_file) + cmd_argv
    else:
        # argv = sys.argv[1:]
        argv = cmd_argv
    args = parser.parse_args(argv)
    #########################################################################
    for key, value in vars(args).items():
        print('Hype-parameter\t{} = {}'.format(key, value))
    #########################################################################
    args = complete_default_train_parser(args)
    logger.info('-' * 100)
    logger.info('Input Argument Information')
    logger.info('-' * 100)
    args_dict = vars(args)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))
    return args

def main(args):
    pl.seed_everything(args.seed)
    #########################################################################
    hgn_model = lightningHGN(args=args)
    return

if __name__ == '__main__':
    ####################################################################################################################
    torch.autograd.set_detect_anomaly(True)
    ####################################################################################################################
    logger.info("IN CMD MODE")
    args = set_args(cmd_argv=sys.argv[1:])