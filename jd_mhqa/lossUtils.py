### Adaptive threhold learning pair-wised loss
"""
References: Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling
References: Focal Loss for Dense Object Detection
References: Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ATLoss(nn.Module):
    """
    Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling
    """
    def __init__(self):
        super().__init__()

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

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

class ATPLoss(nn.Module):
    """
    Adaptive threshold pairwised loss
    """
    def __init__(self):
        super().__init__()



class ATPFLoss(nn.Module):
    """
    Adaptive threshold pair-wised focal loss
    """

class ATWPLoss(nn.Module):
    """
        Adaptive threshold weighted pair-wised  loss
    """