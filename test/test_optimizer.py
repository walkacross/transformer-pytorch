import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_pytorch.trainer.optimizer import NoamOptimizer

model = torch.nn.Linear(1,1)

# Three settings of the lrate hyperparameters.
# the learning rate witin warmup steps is increasing.
opts = [NoamOptimizer(params=model.parameters(), d_model=512, factor=2, warmup_steps=4000), 
        NoamOptimizer(params=model.parameters(), d_model=512, factor=2, warmup_steps=8000),
        NoamOptimizer(params=model.parameters(), d_model=256, factor=2, warmup_steps=4000)]

total_steps = 20000
plt.plot(np.arange(1, total_steps), [[opt.learning_rate(i) for opt in opts] for i in range(1, total_steps)])
plt.legend(["512:4000", "512:8000", "256:4000"])
