import torch
import os
from os.path import join
import numpy as np
import json
from model_envs import MODEL_CLASSES
from csr_mhqa.utils import load_encoder_model, compute_loss, convert_to_tokens
from models.HGN import HierarchicalGraphNetwork
import pytorch_lightning as pl
import torch.nn.functional as F
from eval.hotpot_evaluate_v1 import eval as hotpot_eval
import shutil
from argparse import Namespace
from transformers import AdamW, get_linear_schedule_with_warmup
from csr_mhqa.data_processing import DataHelper

class lightningHGN(pl.LightningModule):
    def __init__(self, args: Namespace):
        super(lightningHGN, self).__init__()
        self.args = args
        cached_config_file = join(self.args.exp_name, 'cached_config.bin')
        if os.path.exists(cached_config_file):
            self.cached_config = torch.load(cached_config_file)
            encoder_path = join(self.args.exp_name, self.cached_config['encoder'])
            model_path = join(self.args.exp_name, self.cached_config['model'])
        else:
            encoder_path = None
            model_path = None
            self.cached_config = None

        _, _, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(self.args.encoder_name_or_path,
                                                    do_lower_case=args.do_lower_case)
        # Set Encoder and Model
        self.encoder, _ = load_encoder_model(self.args.encoder_name_or_path, self.args.model_type)
        self.model = HierarchicalGraphNetwork(config=self.args)

        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

    def prepare_data(self):
        helper = DataHelper(gz=True, config=self.args)
        self.train_data = helper.train_loader
        self.dev_example_dict = helper.dev_example_dict
        self.dev_feature_dict = helper.dev_feature_dict
        self.dev_data = helper.dev_loader

    def setup(self, stage: str):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()
            # Calculate total steps
            if self.args.max_steps > 0:
                self.total_steps = self.args.max_steps
                self.args.num_train_epochs = self.args.max_steps // (
                            len(train_loader) // self.args.gradient_accumulation_steps) + 1
            else:
                self.total_steps = len(train_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.dev_data

    def forward(self, batch):
        inputs = {'input_ids':      batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'token_type_ids': batch['segment_idxs'] if self.args.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids
        batch['context_encoding'] = self.encoder(**inputs)[0]
        batch['context_mask'] = batch['context_mask'].float()
        start, end, q_type, paras, sents, ents, yp1, yp2 = self.model(batch, return_yp=True)
        return start, end, q_type, paras, sents, ents, yp1, yp2

    def training_step(self, batch, batch_idx):
        start, end, q_type, paras, sents, ents, _, _ = self.forward(batch=batch)
        loss_list = compute_loss(self.args, batch, start, end, paras, sents, ents, q_type)
        del batch
        ##################################################################################
        loss, loss_span, loss_type, loss_sup, loss_ent, loss_para = loss_list
        dict_for_progress_bar = {'span_loss': loss_span, 'type_loss': loss_type,
                                 'sent_loss': loss_sup, 'ent_loss': loss_ent,
                                 'para_loss': loss_para}
        dict_for_log = dict_for_progress_bar.copy()
        dict_for_log['step'] = batch_idx + 1
        ##################################################################################
        output = {'loss': loss, 'log': dict_for_log, 'progress_bar': dict_for_progress_bar}
        return output

    def validation_step(self, batch, batch_idx):
        start, end, q_type, paras, sents, ents, yp1, yp2 = self.forward(batch=batch)
        loss_list = compute_loss(self.args, batch, start, end, paras, sents, ents, q_type)
        del batch
        loss, loss_span, loss_type, loss_sup, loss_ent, loss_para = loss_list
        dict_for_log = {'span_loss': loss_span, 'type_loss': loss_type,
                                 'sent_loss': loss_sup, 'ent_loss': loss_ent,
                                 'para_loss': loss_para,
                        'step': batch_idx + 1}
        #######################################################################
        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(self.example_dict, self.feature_dict,
                                                                                    batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        predict_support_np = torch.sigmoid(sents[:, :, 1]).data.cpu().numpy()
        valid_dict = {'answer': answer_dict_, 'ans_type': answer_type_dict_, 'ids': batch['ids'],
                      'ans_type_pro': answer_type_prob_dict_, 'supp_np': predict_support_np}
        #######################################################################
        output = {'valid_loss': loss, 'log': dict_for_log, 'valid_dict_output': valid_dict}
        return output

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in validation_step_outputs]).mean()
        self.log('valid_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        answer_dict = {}
        answer_type_dict = {}
        answer_type_prob_dict = {}

        thresholds = np.arange(0.1, 1.0, 0.025)
        N_thresh = len(thresholds)
        total_sp_dict = [{} for _ in range(N_thresh)]
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        valid_dict_outputs = [x['valid_dict_output'] for x in validation_step_outputs]
        for batch_idx, valid_dict in enumerate(valid_dict_outputs):
            answer_dict_, answer_type_dict_, answer_type_prob_dict_ = valid_dict['answer'], valid_dict['ans_type'], valid_dict['ans_type_pro']
            answer_type_dict.update(answer_type_dict_)
            answer_type_prob_dict.update(answer_type_prob_dict_)
            answer_dict.update(answer_dict_)

            predict_support_np = valid_dict['supp_np']
            batch_ids = valid_dict['ids']

            for i in range(predict_support_np.shape[0]):
                cur_sp_pred = [[] for _ in range(N_thresh)]
                cur_id = batch_ids[i]

                for j in range(predict_support_np.shape[1]):
                    if j >= len(self.example_dict[cur_id].sent_names):
                        break
                    for thresh_i in range(N_thresh):
                        if predict_support_np[i, j] > thresholds[thresh_i]:
                            cur_sp_pred[thresh_i].append(self.example_dict[cur_id].sent_names[j])

                for thresh_i in range(N_thresh):
                    if cur_id not in total_sp_dict[thresh_i]:
                        total_sp_dict[thresh_i][cur_id] = []
                    total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

        def choose_best_threshold(ans_dict, pred_file):
            best_joint_f1 = 0
            best_metrics = None
            best_threshold = 0
            for thresh_i in range(N_thresh):
                prediction = {'answer': ans_dict,
                              'sp': total_sp_dict[thresh_i],
                              'type': answer_type_dict,
                              'type_prob': answer_type_prob_dict}
                tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
                with open(tmp_file, 'w') as f:
                    json.dump(prediction, f)
                metrics = hotpot_eval(tmp_file, self.args.dev_gold_file)
                if metrics['joint_f1'] >= best_joint_f1:
                    best_joint_f1 = metrics['joint_f1']
                    best_threshold = thresholds[thresh_i]
                    best_metrics = metrics
                    shutil.move(tmp_file, pred_file)
            return best_metrics, best_threshold

        output_pred_file = os.path.join(self.args.exp_name, f'pred.epoch_{self.current_epoch + 1}.gpu_{self.trainer.root_gpu}.json')
        output_eval_file = os.path.join(self.args.exp_name, f'eval.epoch_{self.current_epoch + 1}.gpu_{self.trainer.root_gpu}.txt')
        best_metrics, best_threshold = choose_best_threshold(answer_dict, output_pred_file)
        json.dump(best_metrics, open(output_eval_file, 'w'))
        return best_metrics, best_threshold

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]