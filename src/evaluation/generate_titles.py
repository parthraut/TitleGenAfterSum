from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd
import os
from tqdm import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"

def generate_titles_from_articles(directory_path):
    print("-----------------------")
    print("Setting model and tokenizer")

    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    max_input_length = tokenizer.model_max_length

    print("-----------------------")
    print("Loading test set")
    df = pd.read_csv("./data/atn-filtered-publication-section-test2.csv")
    df = df.dropna()
    documents = df["article"].tolist()

    print("-----------------------")
    print("Summarizing")
    batch_size = 128
    all_titles = []
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=max_input_length, truncation=True, padding=True)
        inputs = inputs.to(device)
        titles = model.generate(**inputs, max_length=30, num_beams=1, early_stopping=False)
        batch_titles = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
        all_titles.extend(batch_titles)

    df[model_path] = all_titles
    df.to_csv("./data/atn-filtered-publication-section-test2.csv", index=False)

def generate_summ_titles_from_articles(summ_path, directory_path):
    print("-----------------------")
    print("Setting model and tokenizer")

    items = os.listdir(summ_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(summ_path, item))]
    summ_path = os.path.join(summ_path, subdirectories[-1])

    summ_model = BartForConditionalGeneration.from_pretrained(summ_path)
    summ_model.to(device)
    summ_tokenizer = BartTokenizer.from_pretrained(summ_path)
    summ_max_input_length = summ_tokenizer.model_max_length

    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    title_max_input_length = tokenizer.model_max_length

    print("-----------------------")
    print("Loading test set")
    df = pd.read_csv("./data/atn-filtered-publication-section-test2.csv")
    df = df.dropna()
    documents = df["article"].tolist()

    print("-----------------------")
    print("Summarizing")
    batch_size = 128
    all_titles = []
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        summ_inputs = summ_tokenizer(batch, return_tensors="pt", max_length=summ_max_input_length, truncation=True, padding=True)
        summ_inputs = summ_inputs.to(device)
        summaries = summ_model.generate(**summ_inputs, max_length=200, num_beams=1, early_stopping=False)
        batch_summaries = [summ_tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries]

        inputs = tokenizer(batch_summaries, return_tensors="pt", max_length=title_max_input_length, truncation=True, padding=True)
        inputs = inputs.to(device)
        titles = model.generate(**inputs, max_length=30, num_beams=1, early_stopping=False)
        batch_titles = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
        all_titles.extend(batch_titles)

    df[model_path] = all_titles
    df.to_csv("./data/atn-filtered-publication-section-test2.csv", index=False)

# generate_titles_from_articles("./results/bart/doc2title/all")
# generate_titles_from_articles("./results/bart/doc2title_plus/all")
generate_summ_titles_from_articles("./results/bart/doc2summ", "./results/bart/summ2title/all")



    
