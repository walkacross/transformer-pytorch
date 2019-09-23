import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_pytorch.trainer.metrics import LabelSmoothingLoss

# ================================ #
# 2d input                         #
# ================================ #

# Example of label smoothing.
crit = LabelSmoothingLoss(size=5, label_smoothing=0.4, pad_idx=0)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(predict, torch.LongTensor([2, 1, 0]))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
#

# ==================================== #
# 3d input                             #
# ==================================== #

batch_size = 3
seq_len = 5
dim = 8

input = torch.randn((batch_size,seq_len,dim))
target = torch.tensor(np.random.randint(0,8,(batch_size,seq_len)))
loss = crit(input, target)
print(target.view(-1))
plt.imshow(crit.true_dist)