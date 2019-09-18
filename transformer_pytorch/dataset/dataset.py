import torch
from torchtext import data, datasets
import spacy


spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def get_IWSLT_data_iterator(batch_size=16, max_len=100,root=".data", return_field=False, batch_first=True):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD, batch_first=batch_first)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD,batch_first=batch_first)
    
    MAX_LEN = max_len
    print("start to download data...")
    train_dataset, val_dataset, test_dataset = datasets.IWSLT.splits(root=root,exts=('.de', '.en'), fields=(SRC, TGT), filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train_dataset.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_dataset.trg, min_freq=MIN_FREQ)
    
    print("start to create batch data...")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_dataset, val_dataset, test_dataset),batch_size = batch_size)
    
    if return_field:
        return train_iterator, valid_iterator, test_iterator, SRC, TGT
    else:
        return train_iterator, valid_iterator, test_iterator