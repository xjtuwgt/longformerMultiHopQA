### Adaptive threhold learning pair-wised loss
"""
References: Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling
References: Focal Loss for Dense Object Detection
References: Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def adaptive_threshold_prediction(logits, number_labels=-1, mask = None, type = 'and'):
    if mask is not None:
        logits = logits.masked_fill(mask==0, -1e30)
    th_logits = logits[:, 0].unsqueeze(1)
    output = torch.zeros_like(logits).to(logits)
    out_mask = logits > th_logits
    ######
    if number_labels > 0:
        logits[:, 0] = -1e30
        top_v, _ = torch.topk(logits, k=number_labels, dim=1)
        top_v = top_v[:, -1].unsqueeze(1)
        top_k_mask = logits >= top_v
        if type == 'and':
            out_mask = out_mask & top_k_mask
        elif type == 'or':
            out_mask = out_mask | top_k_mask
        elif type == 'topk':
            out_mask = top_k_mask
        else:
            raise ValueError('mask type {} is not supported'.format(type))
    output[out_mask] = 1.0
    output[:, 0] = output.sum(dim=1)
    return output

class ATLoss(nn.Module):
    """
    Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        # TH label
        """
        :param logits: batch_size * number of labels
        :param labels: batch_size * number of labels (0, 1) matrix, the first column corresponding to the threshold labels
        :return: loss_1 (positive loss), loss_2 (negative loss)
        """
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0 ## threshold based true label information
        labels[:, 0] = 0.0   ## true positive labels

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        ##################################################################
        if self.reduction == 'mean':
            labels_count = labels.sum(1) + 1e-7
            loss1 = loss1/labels_count
        ##################################################################
        # Rank TH to negative classes
        logit2 = logits.masked_fill(n_mask == 0, -1e30)
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

class ATMLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels, mask = None):
        # TH label
        """
        :param logits: batch_size * (number of labels + 1) the first column is for threshold label
        :param labels: batch_size * number of labels (0, 1) matrix, the first column corresponding to the threshold labels
        :return: loss_1 (positive loss), loss_2 (negative loss)
        """
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0  ## threshold based true label information
        labels[:, 0] = 0.0  ## true positive labels

        p_mask = labels + th_label
        n_mask = 1 - labels ## the first column is 1
        ##################################################################
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e30)
        ##################################################################
        # Rank positive classes to TH
        logit1 = logits.masked_fill(p_mask == 0, -1e30)
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        ##################################################################
        if self.reduction == 'mean':
            labels_count = labels.sum(1)
            loss1 = loss1 / labels_count
        ##################################################################
        # Rank TH to negative classes
        logit2 = logits.masked_fill(n_mask == 0, -1e30)
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

class ATPLoss(nn.Module):
    """
    Adaptive threshold pairwised loss
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels, mask = None):
        """
        Pairwised loss
        :param logits:
        :param labels:
        :param mask:
        :return:
        """
        batch_size, label_num = logits.shape
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e30)
        th_logits = logits[:,0].unsqueeze(dim=1).repeat(1, label_num)
        labels[:, 0] = 0.0
        ################################################
        pos_logits = torch.stack([logits, th_logits], dim=-1)
        pos_log = -F.log_softmax(pos_logits, dim=-1)
        loss1 = (pos_log[:,:,0] * labels).sum(1)
        if self.reduction == 'mean':
            labels_count = labels.sum(1) + 1e-7
            loss1 = loss1 / labels_count

        neg_logits = torch.stack([th_logits, logits], dim=-1)
        neg_log = -F.log_softmax(neg_logits, dim=-1)
        n_mask = 1 - labels
        n_mask[:, 0] = 0
        if mask is not None:
            n_mask = n_mask.masked_fill(mask==0, 0)
        loss2 = (neg_log[:,:,0] * n_mask).sum(1)
        if self.reduction == 'mean':
            neg_labels_count = n_mask.sum(1) + 1e-7
            loss2 = loss2/neg_labels_count
        loss = loss1 + loss2
        loss = loss.mean()
        return loss


class ATPFLoss(nn.Module):
    """
    Adaptive threshold pair-wised focal loss
    """

    def __init__(self, reduction: str = 'mean', alpha=1.0,  gamma=2.0):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = 1e-7

    def forward(self, logits, labels, mask=None):
        """
        Pairwised loss
        :param logits:
        :param labels:
        :param mask:
        :return:
        """
        batch_size, label_num = logits.shape
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e30)
        th_logits = logits[:, 0].unsqueeze(dim=1).repeat(1, label_num)
        labels[:, 0] = 0.0
        ################################################
        pos_logits = torch.stack([logits, th_logits], dim=-1)
        loss1 = self.focal_loss(pos_logits, labels)

        neg_logits = torch.stack([th_logits, logits], dim=-1)
        n_mask = 1 - labels
        n_mask[:, 0] = 0
        if mask is not None:
            n_mask = n_mask.masked_fill(mask == 0, 0)
        loss2 = self.focal_loss(neg_logits, n_mask)
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def focal_loss(self, logits, mask):
        logpt = F.log_softmax(logits, dim=-1)
        assert len(logits.shape) == 3
        logpt = logpt[:, :, 0]
        pt = Variable(torch.exp(logpt).to(logits.device))
        pt = torch.clamp(pt, self.smooth, 1.0 - self.smooth)
        loss = -self.alpha * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt * mask
        loss = loss.sum(1)
        if self.reduction == 'mean':
            mask_counts = mask.sum(1) + self.smooth
            loss = loss / mask_counts
        return loss