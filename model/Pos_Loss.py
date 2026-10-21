import torch
from torch import nn

class Pos_loss(nn.Module):
    def __init__(self, class_weight=None):
        super().__init__()
        self.num_label = 13
        if class_weight != None:
            self.class_weight = torch.FloatTensor(class_weight)
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weight)  # reduction='mean'
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()  # reduction='mean'

    def forward(self, span_logits, span_label, span_mask):

        mask_pad = span_mask.view(-1) == 1
        span_label = span_label.view(size=(-1,))[mask_pad]  # (bsz*max_length*max_length,)
        span_logits = span_logits.view(size=(-1, self.num_label))[mask_pad]  # (bsz*max_length*max_length,num_labels)
        span_loss = self.loss_func(input=span_logits, target=span_label)  # (bsz*max_length*max_length,)

        return span_loss
