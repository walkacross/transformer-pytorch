import torch.nn as nn

class TransformerEncoderDecoder(nn.Module):
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        decode_output = self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        return self.generator(decode_output)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class NarrowTransformer(nn.Module):
    def __init__(self, embed, encoder, generator):
        super().__init__()
        self.embed = embed
        self.encoder = encoder
        self.generator = generator
    
    def forward(self, src, src_mask):
        decode_output = self.decode(src, src_mask)
        return self.generator(decode_output)
    
    def decode(self, src, src_mask):
        return self.decoder(self.embed(src), src_mask)