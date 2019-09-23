import torch
import torch.nn as nn
import pdb

class TokenCrossEntropyLoss(nn.Module):
    
    def __init__(self, pad_idx=0):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.loss_function = nn.CrossEntropyLoss(reduction="sum",ignore_index=pad_idx)
    
    def forward(self, input, target):
        
        input_flat = input.contiguous().view(-1, input.size(-1))
        target_flat = target.contiguous().view(-1)
        
        batch_loss_sum = self.loss_function(input_flat, target_flat)
        count = (target_flat != self.pad_idx).sum().item()
        
        return batch_loss_sum, count

class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, label_smoothing, pad_idx):
        assert 0.0 < label_smoothing <=1.0
        super().__init__()
        
        self.pad_idx = pad_idx
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction="sum")
        
        self.smoothing = label_smoothing
        self.condidence = 1.0 - label_smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, input, target):
        #pdb.set_trace()
        input = self.log_softmax(input)
        
        # input flat
        input = input.contiguous().view(-1, input.size(-1))
        target = target.contiguous().view(-1)
        
        true_dist = input.data.clone()
        true_dist.requires_grad = False
        true_dist.fill_(self.smoothing/(self.size-2))
        true_dist.scatter_(1, target.unsqueeze(1), self.condidence)        
        true_dist[:,self.pad_idx] =0
        true_dist.masked_fill_((target==self.pad_idx).unsqueeze(1),0)
        self.true_dist = true_dist 

        loss = self.criterion(input, true_dist)
        count = (target != self.pad_idx).sum().item()

        return loss, count


class AccuracyMetric(nn.Module):
    
    def __init__(self, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
    
    def forward(self, input, target):
        batch_size, seq_len, vocabulary_size = input.size()
        
        input_flat = input.contiguous().view(batch_size*seq_len, vocabulary_size)
        target_flat = target.contiguous().view(batch_size*seq_len)
        
        predicts = input_flat.argmax(dim=1)
        corrects = predicts == target_flat
        corrects.masked_fill_((target_flat==self.pad_idx),0)
        correct_count = corrects.sum().item()
        
        total_count = (target_flat != self.pad_idx).sum().item()
        
        return correct_count, total_count
        