import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward, clones, LayerNorm


class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.size = hidden
    
    def forward(self,x,mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerDecoderBlock(nn.Module):
    
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.size = hidden
        self.self_attention = MultiHeadedAttention(h = attn_heads, d_model=hidden)
        self.src_attention = MultiHeadedAttention(h = attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.middle_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.input_sublayer(x, lambda x: self.self_attention(x,x,x,mask=tgt_mask))
        x = self.middle_sublayer(x, lambda x: self.src_attention(x,m,m,mask=src_mask))
        x = self.output_sublayer(x,self.feed_forward)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)