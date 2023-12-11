from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import os

# setting the seed for random_split
torch.manual_seed(42)

# setting the number of processors for mapping
num_proc = os.cpu_count()
print(f"{num_proc} processors available...")

def load_hf_data(tokenizer, dataset_name, x_key="article", y_key="highlights", return_dl=False, batch_size=32):

    def format_instruction(article, title):
        return f"######\nArticle:\n{article}\n######\Summary:\n{title}"

    def process_dataset(data):
        return {
            x_key: data[x_key],
            y_key: data[y_key],
            "text": format_instruction(data[x_key], data[y_key]) 
        }

    # define tokenize function
    def tokenize(example):
        tokens = tokenizer(example['text'], truncation=True, padding="max_length", max_length=1024)
        filtered_inputs = [
            input_ids for input_ids in tokens["input_ids"] if len(input_ids) <= 1024
        ]
        return {"input_ids": filtered_inputs}
    
    all_datasets = load_dataset(*dataset_name)
    splits = list()
    for split in ["train", "validation", "test"]:
        dataset = all_datasets[split]

        dataset = dataset.map(
            process_dataset, 
            num_proc=num_proc
        )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            # remove_columns=[x_key, y_key, "__index_level_0__"]
            remove_columns=[x_key, y_key, "id"]
        )

        ###########
        tokenized_dict = dict()
        tokenized_dict["input_ids"] = torch.tensor(tokenized_dataset["input_ids"])
        labels = tokenized_dict["input_ids"].clone()
        labels[:, :-1] = tokenized_dict["input_ids"][:, 1:]
        labels[:, -1] = tokenizer.eos_token_id
        tokenized_dict["labels"] = labels
        tokenized_dataset = Dataset.from_dict(tokenized_dict)
        ###########

        splits.append(tokenized_dataset)
    
    # train_ds, val_ds, test_ds = tokenized_dataset["train"], tokenized_dataset["validation"], tokenized_dataset["test"]
    train_ds, val_ds, test_ds = splits[0], splits[1], splits[2]

    if not return_dl:
        return train_ds, val_ds, test_ds
    
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=data_collator)

        return train_dl, val_dl, test_dl

def load_csv_data(tokenizer, csv_fps, x_key="article", y_key="title", chunk_size=1e3, nrows=None, filters=None, return_dl=False, batch_size=32):
    def format_instruction(article, title):
        return f"######\nArticle:\n{article}\n######\nTitle:\n{title}"

    def process_dataset(data):
        return {
            x_key: data[x_key],
            y_key: data[y_key],
            "text": format_instruction(data[x_key], data[y_key]) 
        }

    # define tokenize function
    def tokenize(example):
        tokens = tokenizer(example['text'], truncation=True, padding="max_length", max_length=1024)
        filtered_inputs = [
            input_ids for input_ids in tokens["input_ids"] if len(input_ids) <= 1024
        ]
        return {"input_ids": filtered_inputs}
    
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
            # chunk['gpt2_input'] = chunk.apply(gpt2_map_fn, axis=1)

            hf_dataset = Dataset.from_pandas(chunk)

            # adding text field
            hf_dataset = hf_dataset.map(
                process_dataset, 
                num_proc=num_proc
            )

            tokenized_chunk = hf_dataset.map(
                tokenize,
                batched=True,
                num_proc=num_proc,
                # remove_columns=[x_key, y_key, "__index_level_0__"]
                remove_columns=[x_key, y_key]
            )

            chunk_dict = dict()

            chunk_dict["input_ids"] = torch.tensor(tokenized_chunk["input_ids"])
            labels = chunk_dict["input_ids"].clone()
            labels[:, :-1] = chunk_dict["input_ids"][:, 1:]
            labels[:, -1] = tokenizer.eos_token_id
            chunk_dict["labels"] = labels

            # print(type(chunk_dict["input_ids"]))
            # print(type(chunk_dict["input_ids"][0]))
            # print(type(chunk_dict["input_ids"][0][0]))

            # print(type(chunk_dict["labels"]))
            # print(type(chunk_dict["labels"][0]))
            # print(type(chunk_dict["labels"][0][0]))

            chunk_ds = Dataset.from_dict(chunk_dict)
            tokenized_datasets.append(chunk_ds)

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
