import torch
from torch import nn

class Pos_loss(nn.Module):
    def __init__(self, num_label, class_weight=None):
        super().__init__()
        self.num_label = num_label
        if class_weight != None:
            self.class_weight = torch.FloatTensor(class_weight)
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weight)  # reduction='mean'
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()  # reduction='mean'

    def forward(self, span_logits, span_label, span_mask):

        # 原来的特征处理
        # mask_pad = span_mask.view(-1) == 1
        # span_label = span_label.view(size=(-1,))[mask_pad]  # (bsz*max_length*max_length,)
        # span_logits = span_logits.view(size=(-1, self.num_label))[mask_pad]  # (bsz*max_length*max_length,num_labels)
        # span_loss = self.loss_func(input=span_logits, target=span_label)  # (bsz*max_length*max_length,)
        if not isinstance(span_mask, torch.Tensor):
            span_mask = torch.tensor(span_mask, dtype=torch.bool)
        else:
            span_mask = span_mask.bool()
        span_logits = span_logits.cuda()
        span_label = span_label.cuda()
        span_mask = span_mask.cuda()

        # 2. 使用 contiguous + view 展平
        span_mask_flat = span_mask.contiguous().view(-1)
        span_label_flat = span_label.contiguous().view(-1)
        span_logits_flat = span_logits.contiguous().view(-1, self.num_label)

        # 3. 根据 mask 选择有效元素
        span_label_selected = span_label_flat[span_mask_flat]
        span_logits_selected = span_logits_flat[span_mask_flat]

        # 4. 计算损失
        span_loss = self.loss_func(input=span_logits_selected, target=span_label_selected)

        return span_loss

        return span_loss
