import torch
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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocabulary_size, pad_idx):
        assert 0.0 < label_smoothing <=1.0
        super().__init__()
        
        self.pad_idx = pad_idx
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction="sum")
        
        smoothing_value = label_smoothing / (vocabulary_size - 2)
        smoothed_targets = torch.full((vocabulary_size,), smoothing_value)
        smoothed_targets[self.pad_idx] = 0
        self.register_buffer("smoothed_targets", smoothed_targets.unsqueeze(0))
        
        self.confidence = 1.0 - label_smoothing
    
    def forward(self, input, target):
        
        batch_size, seq_len, vocabulary_size = input.size()

        input_log_softmax = self.log_softmax(input)
        input_flat = input_log_softmax.contiguous().view(batch_size * seq_len, vocabulary_size)
        target_flat = target.contiguous().view(batch_size * seq_len)

        smoothed_target = self.smoothed_targets.repeat(target_flat.size(0), 1).to(target_flat.device)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        smoothed_target.scatter_(1, target_flat.unsqueeze(1), self.confidence)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        smoothed_target.masked_fill_((target_flat == self.pad_idx).unsqueeze(1), 0)
        # masked_targets: (batch_size * seq_len, vocabulary_size)

        loss = self.criterion(input_flat, smoothed_target)
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
        