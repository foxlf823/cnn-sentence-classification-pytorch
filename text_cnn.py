import torch
import torch.autograd as autograd
import torch.utils as utils
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class TextCNN(nn.Module):
    
    def __init__(self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, dropout_keep_prob):

        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedded_chars = nn.Embedding(self.vocab_size, embedding_size)
        
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        
        self.convs = []
        for i, filter_size in enumerate(filter_sizes):
            
            temp = nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(filter_size, embedding_size),
                    )
            self.add_module("Conv_{}".format(i+1), temp)
            self.convs.append(temp)
        
        num_filters_total = num_filters * len(filter_sizes)
        self.linear = nn.Linear(num_filters_total, num_classes)
        
        self.dropout = nn.Dropout(1-dropout_keep_prob)
    
    def forward(self, x):
        """
        x: (batch_size, sequence_length)
        """
        x = self.embedded_chars(x) # (batch_size, sequence_length, embedding_size)
        x = torch.unsqueeze(x, dim=1) # (batch_size, 1, sequence_length, embedding_size)
        
        pooled_outputs = []
        for i, conv in enumerate(self.convs):
            temp = conv(x) # (batch_size, 128, ?, 1)
            temp = F.relu(temp)
            temp = F.max_pool2d(temp, kernel_size=(self.sequence_length - self.filter_sizes[i] + 1, 1))
            pooled_outputs.append(temp) # (batch_size, 128, 1, 1)
          
        hidden = torch.cat(pooled_outputs, dim=1)
        hidden = hidden.view(hidden.size(0), -1)
        
        hidden = self.dropout(hidden)
        
        output = self.linear(hidden)
        
        return output