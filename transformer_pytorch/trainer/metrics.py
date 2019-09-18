import torch.nn as nn

class TokenCrossEntropyLoss(nn.Module):
    
    def __init__(self, pad_idx=0):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.loss_function = nn.CrossEntropyLoss(reduction="sum",ignore_index=pad_idx)
    
    def forward(self, input, target):
        batch_size, seq_len, vocabulary_size = input.size()
        
        input_flat = input.contiguous().view(batch_size*seq_len, vocabulary_size)
        target_flat = target.contiguous().view(batch_size*seq_len)
        
        batch_loss_sum = self.loss_function(input_flat, target_flat)
        count = (target_flat != self.pad_idx).sum().item()
        
        return batch_loss_sum, count


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
        