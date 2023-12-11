from transformers import TextDataset, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments
from dataset_preparation_gpt import load_csv_data, load_hf_data
import torch
from metrics import compute_metrics
from peft import LoraConfig, TaskType, get_peft_model
import os
from torch.nn.parallel import DataParallel
import pandas as pd
import multiprocessing
# from multiprocessing import Pool
from tqdm import tqdm
import torch.multiprocessing as mp
from transformers import pipeline
from torch.multiprocessing import Pool, Process, set_start_method

# setting the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device_count = torch.cuda.device_count()

# setting the number of processors
num_processors = 70

def train_gpt(model_name, batch_size, output_dir, lr, weight_decay, num_epochs, train_ds, val_ds, save_eval_steps=8962, param_efficient=False):
    print("-----------------------")
    print("Setting model and tokenizer")

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("-----------------------")
    print("Setting training arguments")

    modified_batch_size = batch_size // 4
    # training_size = len(train_ds)
    # total_steps = int((training_size / modified_batch_size) * num_epochs)
    # save_eval_steps = total_steps // 10

    # print(f"Total steps: {total_steps}")
    # print(f"save_eval_steps: {save_eval_steps}")

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    # Set training arguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     overwrite_output_dir=True,
    #     num_train_epochs=5,
    #     per_device_train_batch_size=4,
    #     save_steps=1,
    #     save_total_limit=1,
    # )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_steps=save_eval_steps,
        eval_steps=save_eval_steps,
        logging_steps=save_eval_steps,
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=modified_batch_size,
        per_device_eval_batch_size=modified_batch_size,
        gradient_accumulation_steps=4,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        fp16=True,
        load_best_model_at_end=True
    )

    # Train the model
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_ds
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics_lambda,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=output_dir,
    #     evaluation_strategy="steps",
    #     save_steps=save_eval_steps,
    #     eval_steps=save_eval_steps,
    #     logging_steps=save_eval_steps,
    #     save_total_limit=1,
    #     learning_rate=lr,
    #     per_device_train_batch_size=batch_size//device_count,
    #     per_device_eval_batch_size=batch_size//device_count,
    #     gradient_accumulation_steps=4,
    #     weight_decay=weight_decay,
    #     num_train_epochs=num_epochs,
    #     # predict_with_generate=True,
    #     fp16=True,
    #     load_best_model_at_end=True
    # )

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     # compute_metrics=compute_metrics_lambda,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )

    trainer.train()

def train_gpt_doc2title(batch_size, save_name, lr, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training GPT for Doc2Title")

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("-----------------------")
    print("Loading datasets")
    train_ds, val_ds, _ = load_csv_data(
        tokenizer, 
        ("./data/atn-filtered-publication-section-train.csv", "./data/atn-filtered-publication-section-val.csv", "./data/atn-filtered-publication-section-test.csv"),
        chunk_size=chunk_size, 
        nrows=nrows, 
        filters=filters, 
        return_dl=False,
        batch_size=batch_size
    )

    print("-----------------------")
    print("Calling train_gpt")
    train_gpt("gpt2", batch_size, f"results/gpt2/doc2title/{save_name}", lr, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)

def train_gpt_doc2summ(batch_size, lr_summ, weight_decay, num_epochs, param_efficient=False):
    print("-----------------------")
    print("Training GPT for Doc2Summ")

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("-----------------------")
    print("Loading summarization datasets")
    train_ds, val_ds, _ = load_hf_data(
        tokenizer, 
        dataset_name=["cnn_dailymail", "3.0.0"],
        return_dl=False,
        batch_size=batch_size
    )

    print("-----------------------")
    print("Calling train_gpt for summarization")
    train_gpt("gpt2", batch_size, f"results/gpt2/doc2summ", lr_summ, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)

def train_gpt_doc2title_plus(batch_size, save_name, lr_tg, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training GPT for Doc2Title+")

    directory_path = f"results/gpt2/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("-----------------------")
    print("Loading title generation datasets")
    train_ds, val_ds, _ = load_csv_data(
        tokenizer, 
        ("./data/atn-filtered-publication-section-train.csv", "./data/atn-filtered-publication-section-val.csv", "./data/atn-filtered-publication-section-test.csv"),
        chunk_size=chunk_size, 
        nrows=nrows, 
        filters=filters, 
        return_dl=False,
        batch_size=batch_size
    )

    print("-----------------------")
    print("Calling train_gpt")
    train_gpt(model_path, batch_size, f"results/gpt2/doc2title_plus/{save_name}", lr_tg, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)

def summarize_csv(dataset_fp):
    print("-----------------------")
    print("Creating All the News 2.0 Summarized dataset")

    print("-----------------------")
    print("Setting model and tokenizer")

    directory_path = f"results/gpt2/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_input_length = tokenizer.model_max_length

    print("-----------------------")
    print("Reading CSV")
    df = pd.read_csv(dataset_fp, nrows=None)
    df = df.dropna()
    documents = df["article"].tolist()

    print("-----------------------")
    print("Summarizing")
    batch_size = 128
    all_summaries = [None] * len(documents)  
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=max_input_length, truncation=True)
        inputs = [input for input in inputs if len(input) <= 1024]
        inputs = inputs.to(device)
        summaries = model.generate(**inputs, max_length=1024, num_beams=1, early_stopping=False)

        batch_summaries = list()
        for summary in summaries:
            decoded_summary = tokenizer.decode(summary, skip_special_tokens=True)
            decoded_summary = decoded_summary.split("######")[2][10:]

        batch_summaries = [tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries]
        all_summaries.extend(batch_summaries)

    print("-----------------------")
    print("Changing dataframe with summaries")
    df["article"] = all_summaries

    return df

def summarize_all():
    print("-----------------------")
    print("Summarizing train")
    train_summ_df = summarize_csv("data/atn-filtered-publication-section-train.csv")
    train_summ_df.to_csv("data/atn-filtered-publication-section-train-gpt2-summ.csv", index=False)

    print("-----------------------")
    print("Summarizing val")
    val_summ_df = summarize_csv("data/atn-filtered-publication-section-val.csv")
    val_summ_df.to_csv("data/atn-filtered-publication-section-val-gpt2-summ.csv", index=False)

    print("-----------------------")
    print("Summarizing test")
    test_summ_df = summarize_csv("data/atn-filtered-publication-section-test.csv")
    test_summ_df.to_csv("data/atn-filtered-publication-section-test-gpt2-summ.csv", index=False)

def train_gpt_summ2title(batch_size, save_name, lr_tg, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training GPT for SummTitle")

    directory_path = f"results/gpt2/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("-----------------------")
    print("Loading title generation datasets")
    train_ds, val_ds, _ = load_csv_data(
        tokenizer, 
        ("./data/atn-filtered-publication-section-train-gpt2-summ.csv", "./data/atn-filtered-publication-section-val-gpt2-summ.csv", "./data/atn-filtered-publication-section-test-gpt2-summ.csv"),
        chunk_size=chunk_size, 
        nrows=nrows, 
        filters=filters, 
        return_dl=False,
        batch_size=batch_size
    )

    print("-----------------------")
    print("Calling train_gpt")
    train_gpt(model_path, batch_size, f"results/gpt2/doc2title_plus/{save_name}", lr_tg, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)