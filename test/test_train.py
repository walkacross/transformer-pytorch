import torch
import torch.nn as nn
from transformer_pytorch.dataset.dataset import get_IWSLT_data_iterator
from transformer_pytorch.model.model_creator import create_transformer_encoder_decoder
from transformer_pytorch.trainer.train import TransformerTrainer
from transformer_pytorch.trainer.metrics import TokenCrossEntropyLoss, AccuracyMetric
# ===================================== #
# param
# ===================================== #
BATCH_SIZE = 25

data_root = "/media/allen/c54da21a-a3bc-4c5e-a36c-0a41b6108e59/deep_learning/nlp/test/.data"
train_iter, val_iter, test_iter, SRC, TGT = get_IWSLT_data_iterator(batch_size=BATCH_SIZE, root=data_root, return_field=True)
pad_idx = TGT.vocab.stoi["<blank>"]

model = create_transformer_encoder_decoder(src_vocab=len(SRC.vocab),tgt_vocab=len(TGT.vocab), d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1)
#
lr = 0.1 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
loss_function = TokenCrossEntropyLoss(pad_idx=pad_idx)
metric_function = AccuracyMetric(pad_idx=pad_idx)

trainer = TransformerTrainer(model=model,loss_function=loss_function,optimizer=optimizer,metric_function=metric_function, device=device)
trainer.run(dataloader_train=train_iter,pad_idx=pad_idx,dataloader_val=val_iter)