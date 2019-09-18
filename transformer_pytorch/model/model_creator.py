import copy
import torch.nn as nn
import torch.nn.functional as F
from transformer_pytorch.model.embedding.transformer import TransformerEmbedding
from transformer_pytorch.model.transformer_block import TransformerEncoder, TransformerDecoder, TransformerEncoderBlock, TransformerDecoderBlock
from transformer_pytorch.model.transformer import TransformerEncoderDecoder

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


def create_transformer_encoder_decoder(src_vocab, tgt_vocab, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    # create embed layer
    src_embed = TransformerEmbedding(vocab_size = src_vocab, embed_size = d_model)
    tgt_embed = TransformerEmbedding(vocab_size = tgt_vocab, embed_size = d_model)
    
    # create encoder layer
    encoder_layer = TransformerEncoderBlock(hidden=d_model, attn_heads=n_heads, feed_forward_hidden=d_ff, dropout=dropout)
    encoder = TransformerEncoder(encoder_layer, n_layers=n_layers)
    
    # create decoder layer
    decoder_layer = TransformerDecoderBlock(hidden=d_model, attn_heads=n_heads, feed_forward_hidden=d_ff, dropout=dropout)
    decoder = TransformerDecoder(decoder_layer, n_layers=n_layers)
    generator = Generator(d_model=d_model, vocab=tgt_vocab)
    
    # create encoder-decoder model
    model = TransformerEncoderDecoder(encoder=encoder, decoder=decoder, src_embed=src_embed, tgt_embed=tgt_embed, generator=generator)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == "__main__":
    import torch
    import numpy as np
    
    src_vocab = 100
    tgt_vocab  = 150
    batch_size = 3
    src_sequence_max = 6
    tgt_sequence_max = 10
    
    model = create_transformer_encoder_decoder(src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    
    src_input = torch.from_numpy(np.random.randint(1, src_vocab, size=(batch_size, src_sequence_max)))
    tgt_input = torch.from_numpy(np.random.randint(1, tgt_vocab, size=(batch_size, tgt_sequence_max)))
    
    output = model.forward(src=src_input, tgt=tgt_input, src_mask=None, tgt_mask=None)
    assert output.size == torch.Size([batch_size, tgt_sequence_max, tgt_vocab])

