import torch
import numpy as np
import unittest
from transformer_pytorch.model.model_creator import create_transformer_encoder_decoder
import pdb

"""
Many PyTorch operations support NumPy Broadcasting Semantics.

https://pytorch.org/docs/stable/notes/broadcasting.html
"""

attention_score = torch.from_numpy(np.random.randint(0, 1, size=(3, 6, 10)))
attention_mask = torch.from_numpy(np.random.randint(0, 4, size=(3, 1, 10)))
new_score = attention_score.masked_fill(attention_mask==1, 1e3)
 
class TestModelOutput(unittest.TestCase):
    def test_model_output(self):
        src_vocab = 100
        tgt_vocab  = 150
        batch_size = 3
        src_sequence_max = 6
        tgt_sequence_max = 10
    
        model = create_transformer_encoder_decoder(src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    
        src_input = torch.from_numpy(np.random.randint(1, src_vocab, size=(batch_size, src_sequence_max)))
        tgt_input = torch.from_numpy(np.random.randint(1, tgt_vocab, size=(batch_size, tgt_sequence_max)))
        
        output1 = model.forward(src=src_input, tgt=tgt_input, src_mask=None, tgt_mask=None)
        assert output1.shape == torch.Size([batch_size, tgt_sequence_max, tgt_vocab])
        #return output1
        src_mask = torch.from_numpy(np.random.randint(0, 2, size=(batch_size, 1, src_sequence_max)))
        tgt_mask = torch.from_numpy(np.random.randint(0, 2, size=(batch_size, tgt_sequence_max, tgt_sequence_max)))
        #pdb.set_trace()
        output2 = model.forward(src=src_input, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        assert output2.shape == torch.Size([batch_size, tgt_sequence_max, tgt_vocab])
        return output2

if __name__ == "__main__":
    tester = TestModelOutput()
    output = tester.test_model_output()
