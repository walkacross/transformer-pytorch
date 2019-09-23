from transformer_pytorch.dataset.dataset import get_IWSLT_data_iterator
from transformer_pytorch.dataset.utils import rebatch

# ======================================== #
# remember, batch_first                    #
# ======================================== #
# you can set data_root = None

data_root = "/media/allen/c54da21a-a3bc-4c5e-a36c-0a41b6108e59/deep_learning/nlp/test/.data"
train_iter, val_iter, test_iter, SRC, TGT = get_IWSLT_data_iterator(batch_size=3, root=data_root, return_field=True)

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

def print_sequence(sequence, Field, end_word):
    for i in range(1, sequence.size(1)):
        sym = Field.vocab.itos[sequence[0,i]]
        if sym == end_word:
            print("\n")
            break
        print(sym, end=" ")

def show_raw_batch_data(data_iterator):
    for i, batch in enumerate(data_iterator):
        #if batch_first then no transpose
        #pdb.set_trace()
        print("src batch data shape: {}".format(batch.src.shape))
        #print("src batch data: {}".format(batch.src))
        
        print("trg batch data shape: {}".format(batch.trg.shape))
        #print("trg batch data: {}".format(batch.trg))
        
        
        # first example
        the_first_src_example_in_batch = batch.src[:1]
        the_first_trg_example_in_batch = batch.trg[:1]
        
        print_sequence(the_first_src_example_in_batch, SRC, EOS_WORD)
        print_sequence(the_first_trg_example_in_batch, TGT, EOS_WORD)
        if i == 15:
            print("different sequence lenth between diffenent batch")
            break
    
def show_rebatch_data(data_iterator):
    for i, batch in enumerate(data_iterator):
        
        pad_idx = TGT.vocab.stoi["<blank>"]
        batch = rebatch(pad_idx, batch)
        
        print("src batch data shape: {}".format(batch.src.shape))
        print("src batch data: {}".format(batch.src))
        print("src_mask batch data: {}".format(batch.src_mask))
        print("src_mask shape: {}".format(batch.src_mask.shape))
        
        print("trg batch data shape: {}".format(batch.trg.shape))
        print("trg batch data: {}".format(batch.trg))
        print("trg_mask batch data: {}".format(batch.trg_mask))
        print("trg_mask shape: {}".format(batch.trg_mask.shape))
        
        print("trg_y batch data shape: {}".format(batch.trg_y.shape))
        print("trg_y batch data: {}".format(batch.trg_y))
        
        if i == 1:
            print("different sequence lenth between diffenent batch")
            break

def main():
    show_raw_batch_data(train_iter)
    #show_rebatch_data(train_iter)

if __name__ == "__main__":
    main()