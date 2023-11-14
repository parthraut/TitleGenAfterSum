from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from dataset_preparation import load_hf_data, load_csv_data

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096")

train_dl, val_dl, test_dl = load_hf_data(tokenizer, ["cnn_dailymail", "3.0.0"], x_key="article", y_key="highlights", batch_size=32)

csv_fp = './all-the-news-2-1.csv'
train_dl, val_dl, test_dl = load_csv_data(tokenizer, csv_fp, x_key="article", y_key="title", chunk_size=1e5, nrows=None, splits=(0.7, 0.15, 0.15), batch_size=32)

for batch in train_dl:
    print(batch.shape)
    break