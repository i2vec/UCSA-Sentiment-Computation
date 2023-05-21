import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, 
                num_classes,
                hidden_size=768, 
                encode_layer=12,  
                num_filters=3, 
                filter_sizes=[2, 2, 2]):
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.encode_layer=encode_layer

        self.num_filter_total = num_filters * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, num_classes, bias=False)
        self.bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
        ])

    def forward(self, x):
        # x: [bs, seq, hidden]
        x = x.unsqueeze(1) # [bs, channel=1, seq, hidden]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x)) # [bs, channel=1, seq-kernel_size+1, 1]
            mp = nn.MaxPool2d(
                kernel_size = (self.encode_layer-self.filter_sizes[i]+1, 1)
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1) # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes)) # [bs, h=1, w=1, channel=3 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])

        output = self.Weight(h_pool_flat) + self.bias # [bs, n_class]

        return output