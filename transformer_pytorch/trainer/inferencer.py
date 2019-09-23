import torch
from transformer_pytorch.dataset.utils import subsequent_mask
import pdb

def greedy_decode(model, src, src_mask, max_len, start_idx):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1,1).fill_(start_idx).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:,-1])
        _, next_word_idx = torch.max(prob, dim=1)
        next_word_idx = next_word_idx.item()
        ys = torch.cat([ys,torch.ones(1,1).type_as(src.data).fill_(next_word_idx)], dim=1)
    return ys
    

def infer_iterator(model,iterator,SRC,TGT,start_word, end_word, padding_word):
    
    
    def print_sequence(sequence, Field, end_word):
        for i in range(1, sequence.size(1)):
            sym = Field.vocab.itos[sequence[0,i]]
            if sym == end_word:
                print("\n")
                break
            print(sym, end=" ")
    
    
    for i, batch in enumerate(iterator):
        print(i)
        if i == 2:
            break
        # because batch_first, no transpose
        src = batch.src[:1]
        trg = batch.trg[:1]
        
        pad_idx = SRC.vocab.stoi[padding_word]
        start_idx = SRC.vocab.stoi[start_word]
        src_mask = (src != pad_idx).unsqueeze(-2)
        #pdb.set_trace()
        out = greedy_decode(model,src,src_mask,max_len=30,start_idx=start_idx)
        #pdb.set_trace()
        print("Source:", end="\t")
        print_sequence(src,SRC, end_word)
        print("\n")
        
        print("Translation:", end="\t")
        print_sequence(out,TGT,end_word)
        print("\n")
        
        print("Target:", end="\t")
        print_sequence(trg,TGT, end_word)