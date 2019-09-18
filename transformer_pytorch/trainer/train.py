import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from transformer_pytorch.dataset.utils import rebatch


class TransformerTrainer(object):
    def __init__(self, model:nn.Module,
                       loss_function:nn.Module,
                       optimizer:torch.optim.Optimizer,
                       device:torch.device,
                       metric_function:nn.Module) -> None:
        
        self.model = model.to(device)
        self.device = device
        
        self.loss_function = loss_function
        self.metric_funtion = metric_function
        self.optimizer = optimizer
        
        self.log_format = "Epoch: {epoch:>3} \nProgress:{progress:<.1%} \nElapsed: {elapsed} \nTrain_loss:{train_loss} \nVal loss: {val_loss} \nTrain metrics: {train_metrics} \nVal metrics: {val_metrics}"
        
    
    def run_epoch(self, dataloader, mode, pad_idx=0, **kwargs):
        batch_losses = []
        batch_counts = []
        batch_metrics = []
        pad_idx = pad_idx
        
        for i, batch in enumerate(dataloader):
            # 2 data to device
            batch = rebatch(pad_idx, batch)
            src = batch.src.to(self.device)
            src_mask = batch.src_mask.to(self.device)
            trg = batch.trg.to(self.device)
            trg_mask = batch.trg_mask.to(self.device)
            trg_y = batch.trg_y.to(self.device)
            
            #
            output = self.model.forward(src = src, tgt = trg, src_mask=src_mask, tgt_mask=trg_mask)
            batch_loss, batch_count = self.loss_function(output, trg_y)
             
            if mode == "train":
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
            
            batch_losses.append(batch_loss.cpu().item())
            batch_counts.append(batch_count)
            
            batch_correct_count, batch_total_count = self.metric_funtion(output, trg_y)
            assert batch_count ==  batch_total_count
            batch_metrics.append(batch_correct_count)
            batch_counts.append(batch_total_count)
            
            if i % 50 == 0:
                print("batch id: {}".format(i))
                batch_loss = batch_loss.cpu().item()/batch_count
                batch_accuracy = batch_correct_count/batch_total_count
                print("batch_loss:{}, batch_accuracy:{}".format(batch_loss, batch_accuracy))
        
        epoch_loss = sum(batch_losses) / sum(batch_counts)
        epoch_accuracy = sum(batch_metrics) / sum(batch_counts)
        epoch_perplexity = float(np.exp(epoch_loss))
        epoch_metrics = [epoch_loss, epoch_perplexity, epoch_accuracy]
        return epoch_loss, epoch_metrics
    
    
    def run(self, dataloader_train, pad_idx=0, epochs=10, dataloader_val=None):
        self.start_time = datetime.now()
        
        for epoch in range(0,epochs):
            
            self.model.train()
            epoch_start_time = datetime.now()
            train_epoch_loss, train_epoch_metrics = self.run_epoch(dataloader=dataloader_train, mode="train", pad_idx=pad_idx)
            epoch_end_time  = datetime.now()
            
            if dataloader_val is not None:
                self.model.eval()
                val_epoch_loss, val_epoch_metrics = self.run_epoch(dataloader=dataloader_val, mode='val', pad_idx=pad_idx)
            else:
                val_epoch_loss = None
                val_epoch_metrics =[]
            
            
            log_message = self.log_format.format(epoch=epoch,
                                                 progress=epoch / epochs,
                                                 train_loss=train_epoch_loss,
                                                 val_loss=val_epoch_loss,
                                                 train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
                                                 val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
                                                 elapsed=self._elapsed_time())
            print(log_message)
    
    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds
            