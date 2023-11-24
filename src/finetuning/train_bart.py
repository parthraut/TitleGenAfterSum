from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
from dataset_preparation import load_hf_data, load_csv_data
import torch
from metrics import compute_metrics
from peft import LoraConfig, TaskType, get_peft_model
import os
from torch.nn.parallel import DataParallel
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import torch.multiprocessing as mp

# setting the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_count = torch.cuda.device_count()

# setting the number of processors
num_processors = 70

def train_bart(model_name, batch_size, output_dir, lr, weight_decay, num_epochs, train_ds, val_ds, param_efficient=False):
    print("-----------------------")
    print("Setting model and tokenizer")

    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    if param_efficient:
        print("-----------------------")
        print("Using parameter efficient fine-tuning")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=64, 
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ]
        )

        print("-----------------------")
        print("Getting PEFT model")

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print("-----------------------")
    print("Setting training arguments")

    training_size = len(train_ds)
    total_steps = (training_size * num_epochs) / batch_size
    save_eval_steps = total_steps // 10

    # Set up Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_steps=save_eval_steps,
        eval_steps=save_eval_steps,
        logging_steps=save_eval_steps,
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=batch_size//device_count,
        per_device_eval_batch_size=batch_size//device_count,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    compute_metrics_lambda = lambda x: compute_metrics(tokenizer, x)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics_lambda,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("-----------------------")
    print("Training the model")

    trainer.train()

def train_bart_doc2title(batch_size, save_name, lr, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training BART for Doc2Title")

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

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
    print("Calling train_bart")
    train_bart("facebook/bart-base", batch_size, f"results/bart/doc2title/{save_name}", lr, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)

def train_bart_doc2summ(batch_size, lr_summ, weight_decay, num_epochs, param_efficient=False):
    print("-----------------------")
    print("Training BART for Doc2Summ")

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    print("-----------------------")
    print("Loading summarization datasets")
    train_ds, val_ds, _ = load_hf_data(
        tokenizer, 
        dataset_name=["cnn_dailymail", "3.0.0"],
        return_dl=False,
        batch_size=batch_size
    )

    print("-----------------------")
    print("Calling train_bart for summarization")
    train_bart("facebook/bart-base", batch_size, f"results/bart/doc2summ", lr_summ, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)

def train_bart_doc2title_plus(batch_size, save_name, lr_tg, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training BART for Doc2Title+")

    directory_path = f"results/bart/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = BartTokenizer.from_pretrained(model_path)

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
    print("Calling train_bart")
    train_bart(model_path, batch_size, f"results/bart/doc2title_plus/{save_name}", lr_tg, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)\
    
def summarize(args):
    model_path, value = args

    model = BartForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    max_input_tokens = model.config.max_position_embeddings
    tokenizer = BartTokenizer.from_pretrained(model_path)

    input_ids = tokenizer(value, return_tensors="pt").input_ids
    input_ids = input_ids[:, :max_input_tokens]
    input_ids = input_ids.to(device)

    # Generate output
    output_ids = model.generate(input_ids, max_length=max_input_tokens)

    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

# def summarize2(model, tokenizer, max_input_tokens, values):
#     output_texts = list()
#     for value in values:
#         input_ids = tokenizer(value, return_tensors="pt").input_ids
#         input_ids = input_ids[:, :max_input_tokens]
#         # input_ids = input_ids.to(device)

#         # Generate output
#         output_ids = model.generate(input_ids, max_length=max_input_tokens)

#         # Decode the output
#         output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         output_texts.append(output_text)

#     return output_texts

def summarize_csv(dataset_fp):
    print("-----------------------")
    print("Creating All the News 2.0 Summarized dataset")

    print("-----------------------")
    print("Setting model and tokenizer")

    directory_path = f"results/bart/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    print("-----------------------")
    print("Reading CSV")
    df = pd.read_csv(dataset_fp, nrows=None)
    df = df.dropna()

    print("-----------------------")
    print("Generating summarization inputs")
    inputs = list()
    for article in tqdm(df["article"], total=len(df["article"])):
        inputs.append((model_path, article))

    print("-----------------------")
    print("Summarizing")
    summaries = list()
    for input in tqdm(inputs):
        summary = summarize(input)
        summaries.append(summary)

    print("-----------------------")
    print("Changing dataframe with summaries")
    df["article"] = summaries

    return df

def summarize_all():
    # print("-----------------------")
    # print("Summarizing train")
    # train_summ_df = summarize_csv("data/atn-filtered-publication-section-train.csv")
    # train_summ_df.to_csv("data/atn-filtered-publication-section-train-bart-summ.csv", index=False)

    # print("-----------------------")
    # print("Summarizing val")
    # val_summ_df = summarize_csv("data/atn-filtered-publication-section-val.csv")
    # val_summ_df.to_csv("data/atn-filtered-publication-section-val-bart-summ.csv", index=False)

    print("-----------------------")
    print("Summarizing test")
    test_summ_df = summarize_csv("data/atn-filtered-publication-section-test.csv")
    test_summ_df.to_csv("data/atn-filtered-publication-section-test-bart-summ.csv", index=False)

def train_bart_summ2title(batch_size, save_name, lr_tg, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training BART for SummTitle")

    directory_path = f"results/bart/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = BartTokenizer.from_pretrained(model_path)

    print("-----------------------")
    print("Loading title generation datasets")
    train_ds, val_ds, _ = load_csv_data(
        tokenizer, 
        ("./data/atn-filtered-publication-section-train-summ.csv", "./data/atn-filtered-publication-section-val-summ.csv", "./data/atn-filtered-publication-section-test-summ.csv"),
        chunk_size=chunk_size, 
        nrows=nrows, 
        filters=filters, 
        return_dl=False,
        batch_size=batch_size
    )

    print("-----------------------")
    print("Calling train_bart")
    train_bart(model_path, batch_size, f"results/bart/doc2title_plus/{save_name}", lr_tg, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)

# train_bart_doc2summ(
#     batch_size=16, 
#     lr_summ=8e-4, 
#     weight_decay=1e-1, 
#     num_epochs=2, 
#     param_efficient=True
# )

# train_bart_doc2title_plus(
#     batch_size=16, 
#     save_name="tmz", 
#     lr_tg=8e-5, 
#     weight_decay=1e-1, 
#     num_epochs=2, 
#     chunk_size=1e4, 
#     nrows=None, 
#     filters=[("publication", "TMZ")],
#     param_efficient=True
# )

summarize_all()