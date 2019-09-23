import os
import torch
from transformer_pytorch.model.model_creator import create_transformer_encoder_decoder
from transformer_pytorch.trainer.inferencer import greedy_decode
import pdb

def findpth(path):
    ret = []
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(".pth"):
                ret.append(de_path)
    return ret

d_model = 512
n_layers = 6
n_heads = 8
d_ff = 2048
dropout =0.1
epochs = 100

def test_greedy_decode():
    model = create_transformer_encoder_decoder(src_vocab=1000,tgt_vocab=1000, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
    model.eval()

    src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])
    src_mask = torch.ones(1, 1, 10)
    out = greedy_decode(model, src, src_mask, max_len=10, start_idx=1)
    print(out)

# ======================================================== #
# test inference                                           #
# ======================================================== #
from transformer_pytorch.dataset.dataset import get_IWSLT_data_iterator
from transformer_pytorch.trainer.inferencer import infer_iterator


def test_infer_dataloader():

    BATCH_SIZE = 16
    data_root = "/media/allen/c54da21a-a3bc-4c5e-a36c-0a41b6108e59/deep_learning/nlp/test/.data"
    train_iter, val_iter, test_iter, SRC, TGT = get_IWSLT_data_iterator(batch_size=BATCH_SIZE, root=data_root, return_field=True)
    
    # create model 
    model = create_transformer_encoder_decoder(src_vocab=len(SRC.vocab),tgt_vocab=len(TGT.vocab), d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
    #device = torch.device('cpu')
    #path = os.getcwd()
    #model_path_list = findpth(path)    
    #if len(model_path_list):
    #    model.load_state_dict(torch.load(model_path_list[0]), map_location=device)
    model.eval()
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    infer_iterator(model,iterator=val_iter,SRC=SRC,TGT=TGT,start_word=BOS_WORD, end_word=EOS_WORD, padding_word=BLANK_WORD)
