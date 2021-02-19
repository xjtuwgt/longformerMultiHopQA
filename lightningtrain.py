import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import logging
import torch
import sys, os
from utils.gpu_utils import gpu_setting
from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from lightningmodel.lightningHGN import lightningHGN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def gpu_id_setting(args):
    if torch.cuda.is_available():
        if args.gpus > 0:
            free_gpu_ids, used_memory = gpu_setting(num_gpu=args.gpus)
            logging.info('{} gpus with used memory = {}, gpu ids = {}'.format(len(free_gpu_ids), used_memory, free_gpu_ids))
            if args.gpus > len(free_gpu_ids):
                gpu_list_str = ','.join([str(_) for _ in free_gpu_ids])
                args.gpus = len(free_gpu_ids)
            else:
                gpu_list_str = ','.join([str(free_gpu_ids[i]) for i in range(args.gpus)])
            args.gpu_list = gpu_list_str
            logging.info('gpu list = {}'.format(gpu_list_str))
    return args

def trainer_builder(args):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info("PyTorch Lighting Trainer constructing...")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.exp_name)
    ####################################################################################################################
    check_point_dir = args.exp_name
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                          mode='min',
                                          save_top_k=-1,
                                          dirpath=check_point_dir,
                                          filename='HGN_hotpotQA-{epoch:02d}-{valid_loss:.4f}')
    ####################################################################################################################
    if args.gpus > 0:
        gpu_list_str = args.gpu_list
        gpu_ids = [int(x) for x in gpu_list_str.split(',')]
        trainer = pl.Trainer(logger=tb_logger,
                             gradient_clip_val=args.max_grad_norm,
                             gpus=gpu_ids,
                             val_check_interval=args.val_check_interval,
                             accumulate_grad_batches=args.gradient_accumulation_steps,
                             callbacks=[checkpoint_callback],
                             accelerator=args.accelerator,
                             precision=args.precision,
                             plugins=args.plugins,
                             log_every_n_steps=args.logging_steps,
                             max_epochs=int(args.num_train_epochs))
    else:
        trainer = pl.Trainer(logger=tb_logger,
                             gradient_clip_val=args.max_grad_norm,
                             val_check_interval=args.val_check_interval,
                             accumulate_grad_batches=args.gradient_accumulation_steps,
                             log_every_n_steps=args.logging_steps,
                             max_epochs=int(args.num_train_epochs))
    return trainer


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
    args = complete_default_train_parser(args)
    #######
    args = gpu_id_setting(args)
    #######
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
    model = lightningHGN(args=args)
    model.prepare_data()
    model.setup()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    logging.info('*' * 75)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ####################################################################################################################
    trainer = trainer_builder(args=args)
    ####################################################################################################################
    return trainer, model

if __name__ == '__main__':
    ####################################################################################################################
    torch.autograd.set_detect_anomaly(True)
    ####################################################################################################################
    logger.info("IN CMD MODE")
    args = set_args(cmd_argv=sys.argv[1:])
    trainer, model = main(args=args)
    trainer.fit(model=model)