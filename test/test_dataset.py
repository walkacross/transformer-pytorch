from transformer_pytorch.dataset.dataset import get_IWSLT_data_iterator
from transformer_pytorch.dataset.utils import rebatch

# you can set data_root = None
data_root = "/media/allen/c54da21a-a3bc-4c5e-a36c-0a41b6108e59/deep_learning/nlp/test/.data"
train_iter, val_iter, test_iter, SRC, TGT = get_IWSLT_data_iterator(batch_size=3, root=data_root, return_field=True)


def show_raw_batch_data(data_iterator):
    for i, batch in enumerate(data_iterator):
        print("src batch data shape: {}".format(batch.src.shape))
        print("src batch data: {}".format(batch.src))
        
        print("trg batch data shape: {}".format(batch.trg.shape))
        print("trg batch data: {}".format(batch.trg))
        
        if i == 1:
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
    show_rebatch_data(train_iter)

if __name__ == "__main__":
    main()