from transformers import AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments
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

# setting the number of processors
num_processors = 70

def train_gpt(model_name, batch_size, output_dir, lr, weight_decay, num_epochs, train_ds, val_ds, param_efficient=False):
    print("-----------------------")
    print("Setting model and tokenizer")

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # config = AutoConfig.from_pretrained(
    #     model_name,
    #     vocab_size=len(tokenizer),
    #     n_ctx=1024,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )

    # model = GPT2LMHeadModel(config)
    # model.to(device)

    # if param_efficient:
    #     print("-----------------------")
    #     print("Using parameter efficient fine-tuning")

    #     peft_config = LoraConfig(
    #         task_type=TaskType.SEQ_2_SEQ_LM, 
    #         inference_mode=False, 
    #         r=16, 
    #         lora_alpha=64, 
    #         lora_dropout=0.1,
    #         target_modules=[
    #             "q_proj",
    #             "k_proj",
    #             "v_proj",
    #             "o_proj"
    #         ]
    #     )

    #     print("-----------------------")
    #     print("Getting PEFT model")

    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()

    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token='<PAD>')

    print("-----------------------")
    print("Setting training arguments")

    training_size = len(train_ds)
    total_steps = (training_size * num_epochs) / batch_size
    save_eval_steps = total_steps // 10

    # # Set up TrainingArguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     evaluation_strategy="steps",
    #     save_steps=save_eval_steps,
    #     eval_steps=save_eval_steps,
    #     logging_steps=save_eval_steps,
    #     save_total_limit=1,
    #     learning_rate=lr,
    #     per_device_train_batch_size=batch_size//device_count,
    #     per_device_eval_batch_size=batch_size//device_count,
    #     weight_decay=weight_decay,
    #     num_train_epochs=num_epochs,
    #     fp16=True,
    #     load_best_model_at_end=True
    # )

    # # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # compute_metrics_lambda = lambda x: compute_metrics(tokenizer, x)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     compute_metrics=compute_metrics_lambda,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     tokenizer=tokenizer,
    #     # data_collator=data_collator,
    # )


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        save_steps=2,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
    )

    # tokenizer.pad_token = tokenizer.eos_token
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # args = TrainingArguments(
    #     output_dir=output_dir,
    #     evaluation_strategy="steps",
    #     save_steps=save_eval_steps,
    #     eval_steps=save_eval_steps,
    #     logging_steps=save_eval_steps,
    #     save_total_limit=1,
    #     learning_rate=lr,
    #     per_device_train_batch_size=batch_size//device_count,
    #     per_device_eval_batch_size=batch_size//device_count,
    #     weight_decay=weight_decay,
    #     num_train_epochs=num_epochs,
    #     fp16=True,
    #     load_best_model_at_end=True
    # )

    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=args,
    #     data_collator=data_collator,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds
    # )

    print("-----------------------")
    print("Training the model")

    trainer.train()

def train_gpt_doc2title(batch_size, save_name, lr, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training GPT for Doc2Title")

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
    train_gpt("gpt2-medium", batch_size, f"results/gpt2/doc2title/{save_name}", lr, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)

def train_gpt_doc2summ(batch_size, lr_summ, weight_decay, num_epochs, param_efficient=False):
    print("-----------------------")
    print("Training GPT for Doc2Summ")

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
    train_gpt(model_path, batch_size, f"results/gpt2/doc2title_plus/{save_name}", lr_tg, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)\

# def summarize(args):
#     model, max_input_tokens, tokenizer, article = args

#     input_ids = tokenizer(article, return_tensors="pt").input_ids
#     input_ids = input_ids[:, :max_input_tokens]
#     input_ids = input_ids.to(device)

#     # Generate output
#     output_ids = model.generate(input_ids, max_length=200)

#     # Decode the output
#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     return output_text

# def summarize2(args):
#     reader, value = args
#     return reader(value)

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
    all_summaries = []
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=max_input_length, truncation=True)
        inputs = inputs.to(device)
        summaries = model.generate(**inputs, max_length=150, num_beams=1, early_stopping=False)
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

# summarize_all()

train_gpt_doc2title(
    batch_size=16, 
    save_name="all", 
    lr=8e-4, 
    weight_decay=1e-1, 
    num_epochs=5, 
    chunk_size=10, 
    nrows=100, 
    filters=None,
    param_efficient=False
)