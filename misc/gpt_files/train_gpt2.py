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
device_count = torch.cuda.device_count()

def fine_tune_gpt2(model_name, train_file, output_dir):
    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=2)
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=1,
        save_total_limit=1,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

fine_tune_gpt2("gpt2", "train_file.txt", "./gpt2-finetuned/")