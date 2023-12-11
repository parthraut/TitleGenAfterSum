from transformers import GPT2LMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_metric
import pickle
import pandas as pd

rouge_metric = load_metric("rouge")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=500)
test_df = test_df[["article", "title"]]
test_df = test_df.dropna()
test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 1 and len(tokenizer(x)["input_ids"]) <= 256)]

# test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=None)
# test_df = test_df[["article", "title"]]
# test_df = test_df.dropna()
# test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 257 and len(tokenizer(x)["input_ids"]) <= 512)]

doc2title_titles = list()
doc2title_plus_titles = list()
summ2title_titles = list()

def evaluate_single_model(model_path, model_architecture):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)

    test_documents = test_df["article"].tolist() 
    test_titles = test_df["title"].tolist() 
        

    batch_size = 25

    titles = None

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    for i in tqdm(range(0, 25, batch_size)):
        batch = test_documents[i:i + batch_size]
        inputs = list()
        attention_masks = list()
        for doc in batch:
            tokens = tokenizer(doc, return_tensors="pt")
            inputs.append(tokens["input_ids"][0])
            attention_masks.append(tokens["attention_mask"][0])

        filtered_inputs = list()
        # filtered_attention_masks = list()
        filtered_gt_titles = list()
        for j in range(len(inputs)):
            if len(inputs[j]) <= 1024:
                padding_size = max(0, 1024 - len(inputs[j]))

                inputs[j] = torch.cat([inputs[j], torch.tensor([pad_token_id] * padding_size)])
               
                filtered_inputs.append(inputs[j])
                # filtered_attention_masks.append(attention_masks[j])
                filtered_gt_titles.append(test_titles[i+j])

        filtered_inputs = torch.cat([tokens.unsqueeze(0) for tokens in filtered_inputs], dim=0)
        filtered_inputs = filtered_inputs.to(device).long()

        # filtered_attention_masks = torch.cat([tokens.unsqueeze(0) for tokens in filtered_attention_masks], dim=0)
        # filtered_attention_masks = filtered_attention_masks.to(device).long()

        titles = model.generate(filtered_inputs, max_length=30, num_beams=1, early_stopping=False)

        decoded_titles = list()
        for title in titles:
            output_text = tokenizer.decode(title, skip_special_tokens=True)
            decoded_titles.append(output_text)

    return decoded_titles, batch

def evaluate_double_model(summ_model_path, tg_model_path, model_architecture):
    tokenizer = BartTokenizer.from_pretrained(model_path)

    # test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=None)
    # test_df = test_df[["article", "title"]]
    # test_df = test_df.dropna()
    # test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 769 and len(tokenizer(x)["input_ids"]) <= 1024)]

    # test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=None)
    # test_df = test_df[["article", "title"]]
    # test_df = test_df.dropna()
    # test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 769 and len(tokenizer(x)["input_ids"]) <= 1024)]
        
    test_documents = test_df["article"].tolist() 
    test_titles = test_df["title"].tolist() 
    
    tokenizer = BartTokenizer.from_pretrained(tg_model_path)

    summ_model = BartForConditionalGeneration.from_pretrained(summ_model_path)
    summ_model.to(device)

    tg_model = BartForConditionalGeneration.from_pretrained(tg_model_path)
    tg_model.to(device)

    batch_size = 25

    titles = None

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    for i in tqdm(range(0, batch_size, batch_size)):
        batch = test_documents[i:i + batch_size]
        inputs = list()
        attention_masks = list()
        for doc in batch:
            tokens = tokenizer(doc, return_tensors="pt")
            inputs.append(tokens["input_ids"][0])
            attention_masks.append(tokens["attention_mask"][0])

        filtered_inputs = list()
        # filtered_attention_masks = list()
        filtered_gt_titles = list()
        for j in range(len(inputs)):
            if len(inputs[j]) <= 1024:
                padding_size = max(0, 1024 - len(inputs[j]))

                inputs[j] = torch.cat([inputs[j], torch.tensor([pad_token_id] * padding_size)])

                filtered_inputs.append(inputs[j])
                # filtered_attention_masks.append(attention_masks[j])
                filtered_gt_titles.append(test_titles[i+j])

        filtered_inputs = torch.cat([tokens.unsqueeze(0) for tokens in filtered_inputs], dim=0)
        filtered_inputs = filtered_inputs.to(device).long()

        # filtered_attention_masks = torch.cat([tokens.unsqueeze(0) for tokens in filtered_attention_masks], dim=0)
        # filtered_attention_masks = filtered_attention_masks.to(device).long()

        summaries = summ_model.generate(filtered_inputs, max_length=500, num_beams=1, early_stopping=False)
        batch_summaries = [tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries]
        inputs = tokenizer(batch_summaries, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        inputs = inputs.to(device)
        titles = tg_model.generate(**inputs, max_length=50, num_beams=1, early_stopping=False)
        # titles = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
        decoded_titles = list()
        for i in range(len(titles)):
            output_text = tokenizer.decode(titles[i], skip_special_tokens=True)
            decoded_titles.append(output_text)

    return decoded_titles, batch

        


model_path = "results/bart/doc2title/all/checkpoint-44810"
doc2title_titles, batch = evaluate_single_model(model_path, "bart")

model_path = "results/bart/doc2title_plus/all/checkpoint-44810"
doc2title_plus_titles, _ = evaluate_single_model(model_path, "bart")

tg_model_path = "results/bart/summ2title/all/checkpoint-32376"
summ_model_path = "results/bart/doc2summ/checkpoint-30000"
summ2title_titles, _ = evaluate_double_model(summ_model_path, tg_model_path, "bart")

df = pd.DataFrame({
    "article": batch,
    "doc2title": doc2title_titles,
    "doc2title+": doc2title_plus_titles,
    "summ2title": summ2title_titles
})

df.to_csv('human_eval_dataset.csv', index=False)
