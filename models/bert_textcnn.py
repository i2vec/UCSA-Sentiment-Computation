import torch
import torch.nn as nn
from transformers import AutoModel
from .textcnn import TextCNN


class BertTextCNN(nn.Module):
    def __init__(self, 
                 num_classes=7, 
                 loss_fn=torch.nn.CrossEntropyLoss(reduction='mean')):
        super(BertTextCNN, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True, return_dict=True)
        self.loss_fn = loss_fn
        for name, param in self.bert.named_parameters():
            if not 'pooler' in name:
                param.requires_grad_(False)
        self.textcnn = TextCNN(num_classes)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # 返回一个output字典
        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]
        hidden_states = outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        logits = self.textcnn(cls_embeddings)
        return {
            'logits': logits, 
            'loss': self.loss_fn(logits, labels)
        }