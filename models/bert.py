import torch
import torch.nn as nn
from transformers import AutoModel


class BertForClassification(nn.Module):
    def __init__(self, 
                 num_classes=7,
                 loss_fn=torch.nn.CrossEntropyLoss(reduction='mean')):
        super(BertForClassification, self).__init__()
        self.loss_fn = loss_fn
        self.bert = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True, return_dict=True)
        for name, param in self.bert.named_parameters():
            if not 'pooler' in name:
                param.requires_grad_(False)
        self.fc_out = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # 返回一个output字典
        cls = outputs['hidden_states'][-1][:, 0, :].squeeze(1)
        logits = self.fc_out(cls)
        return {
            'logits': logits, 
            'loss': self.loss_fn(logits, labels)
        }
        
        