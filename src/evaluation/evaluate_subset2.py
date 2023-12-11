from transformers import GPT2LMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_metric
import matplotlib.pyplot as plt
import pickle

rouge_metric = load_metric("rouge")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=1000)
test_df = test_df.dropna()

def evaluate_single_model(model_path):
    test_documents = test_df["article"].tolist() 
    test_documents = [f"######\nArticle:\n{doc}\n######\nTitle:\n" for doc in test_documents]
    test_titles = test_df["title"].tolist() 

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    pred_titles = list()
    gt_titles = list()

    for i in tqdm(range(len(test_documents))):
        input_text = test_documents[i]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        if len(input_ids[0]) > 1024:
            continue 
        
        input_ids = input_ids[:, :1024]
        input_ids = input_ids.to(device)

        # Generate output
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output_ids = model.generate(input_ids, max_length=1024, attention_mask=attention_mask, repetition_penalty=2.0, num_return_sequences=1)

        # Decode the output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred_titles.append(output_text.split("######")[2][8:])
        gt_titles.append(test_titles[i])

    print(pred_titles[0])
    print(gt_titles[0])

    rouge_scores = rouge_metric.compute(
        predictions=pred_titles,
        references=gt_titles,
    )

    mid_fmeasure_values = {key: value.mid.fmeasure for key, value in rouge_scores.items()}
    print(mid_fmeasure_values)

    with open(f'gpt2lengths_{model_path.replace("/", "_")}.pkl', 'wb') as file:
        pickle.dump([len(p.split()) for p in pred_titles], file)

model_path = "results/gpt2/doc2title/all/checkpoint-17924"
evaluate_single_model(model_path)
# model_path = "results/gpt2/doc2title_plus/all/checkpoint-17924"
# evaluate_single_model(model_path)
        