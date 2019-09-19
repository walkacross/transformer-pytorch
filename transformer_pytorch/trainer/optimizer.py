from torch.optim import Adam

class NoamOptimizer(Adam):
    
    def __init__(self, params, d_model,factor=2, warmup_steps=4000, betas=(0.9,0.98), eps=1e-9):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr = 0
        self.step_num = 0
        self.factor = factor
        
        super(NoamOptimizer, self).__init__(params, betas=betas, eps=eps)
    
    def step(self):
        self.step_num += 1
        self.lr = self.learning_rate()
        for group in self.param_groups:
            group["lr"] = self.lr
        super(NoamOptimizer, self).step()
    
    def learning_rate(self, step=None):
        if step is None:
            step = self.step_num
        return self.factor * self.d_model**(-0.5) * min(step ** (-0.5), step * self.warmup_steps**(-1.5))
