from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq
import pandas as pd
from torch.utils.data import DataLoader, random_split

num_proc = 70

def load_hf_data(tokenizer, dataset_name, x_key="article", y_key="highlights", return_dl=False, batch_size=32):

    # define tokenize function
    def tokenize(example):
        tokenized_x = tokenizer(example[x_key], truncation=True)
        tokenized_y = tokenizer(example[y_key], truncation=True)

        return {
            "input_ids": tokenized_x["input_ids"],
            "labels": tokenized_y["input_ids"],
        }
    
    dataset = load_dataset(*dataset_name)
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=[x_key, y_key, "id"]
    )

    train_ds, val_ds, test_ds = tokenized_dataset["train"], tokenized_dataset["validation"], tokenized_dataset["test"]

    if not return_dl:
        return train_ds, val_ds, test_ds
    
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)

        return train_dl, val_dl, test_dl

def split_dataset(dataset, splits):
    train_split, val_split, test_split = splits
    assert train_split >= 0 and val_split >= 0 and test_split >= 0
    assert abs(sum([train_split, val_split, test_split]) - 1) < 1e-6

    shuffled_dataset = dataset.shuffle()
    total_size = len(shuffled_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(shuffled_dataset, [train_size, val_size, test_size])
    return train_set, val_set, test_set

def load_csv_data(tokenizer, csv_fp, x_key="article", y_key="title", chunk_size=1e3, nrows=None, splits=(0.7, 0.2, 0.1), return_dl=False, batch_size=32):

    # define tokenize function
    def tokenize(example):
        tokenized_x = tokenizer(example[x_key], truncation=True)
        tokenized_y = tokenizer(example[y_key], truncation=True)

        return {
            "input_ids": tokenized_x["input_ids"],
            "labels": tokenized_y["input_ids"],
        }

    tokenized_datasets = []

    # read chunks of the csv to avoid loading it all at once
    if nrows == None:
        i = 0
        total = int(2.7*10**6//chunk_size)

        for chunk in pd.read_csv(csv_fp, chunksize=chunk_size):
            print(f"Chunk {i + 1} / {total}")
            chunk = chunk[[x_key, y_key]]
            chunk = chunk.dropna()

            hf_dataset = Dataset.from_pandas(chunk)
            tokenized_chunk = hf_dataset.map(
                tokenize,
                batched=True,
                num_proc=num_proc,
                remove_columns=[x_key, y_key, "__index_level_0__"]
            )
            
            tokenized_datasets.append(tokenized_chunk)
            i += 1

    else:
        i = 0
        total = int(nrows//chunk_size)

        for chunk in pd.read_csv(csv_fp, chunksize=chunk_size, nrows=nrows):
            print(f"Chunk {i + 1} / {total}")
            chunk = chunk[[x_key, y_key]]
            chunk = chunk.dropna()

            hf_dataset = Dataset.from_pandas(chunk)
            tokenized_chunk = hf_dataset.map(
                tokenize,
                batched=True,
                num_proc=num_proc,
                remove_columns=[x_key, y_key, "__index_level_0__"]
            )

            tokenized_datasets.append(tokenized_chunk)
            i += 1

    # concatenate tokenized dataset chunks
    tokenized_dataset = concatenate_datasets(tokenized_datasets)
    train_ds, val_ds, test_ds = split_dataset(tokenized_dataset, splits)

    if not return_dl:
        return train_ds, val_ds, test_ds
    
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)

        return train_dl, val_dl, test_dl

