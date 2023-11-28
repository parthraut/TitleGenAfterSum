from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

# setting the seed for random_split
torch.manual_seed(42)

# setting the number of processors for mapping
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

def load_csv_data(tokenizer, csv_fps, x_key="article", y_key="title", chunk_size=1e3, nrows=None, filters=None, return_dl=False, batch_size=32):
    
    # define tokenize function
    def tokenize(example):
        tokenized_x = tokenizer(example[x_key], truncation=True)
        tokenized_y = tokenizer(example[y_key], truncation=True)

        return {
            "input_ids": tokenized_x["input_ids"],
            "labels": tokenized_y["input_ids"],
        }

    dataset_splits = []
    for csv_fp in csv_fps:
        print("-----------")
        print(f"Loading {csv_fp}")

        tokenized_datasets = []

        # read chunks of the csv to avoid loading it all at once
        for chunk in pd.read_csv(csv_fp, chunksize=chunk_size, nrows=nrows):
            # filter chunks based on provided values
            if not filters == None:
                for col, val in filters:
                    chunk = chunk[chunk[col] == val]
            
            # keep only x and y columns & delete bad rows
            chunk = chunk[[x_key, y_key]]
            chunk = chunk.dropna()

            hf_dataset = Dataset.from_pandas(chunk)
            tokenized_chunk = hf_dataset.map(
                tokenize,
                batched=True,
                num_proc=num_proc,
                # remove_columns=[x_key, y_key, "__index_level_0__"]
                remove_columns=[x_key, y_key]
            )

            tokenized_datasets.append(tokenized_chunk)

        # concatenate tokenized dataset chunks
        tokenized_dataset = concatenate_datasets(tokenized_datasets)
        dataset_splits.append(tokenized_dataset)
        
    train_ds, val_ds, test_ds = tuple(dataset_splits)

    if not return_dl:
        return train_ds, val_ds, test_ds
    
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)

        return train_dl, val_dl, test_dl
