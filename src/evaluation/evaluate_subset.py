from transformers import GPT2LMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_metric
import pickle

rouge_metric = load_metric("rouge")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# test_df = test_df[test_df["section"] == "sports"]

# "politics"

# condition_column_5 = "section"
# condition_value_5 = "Financials"

# condition_column_6 = "section"
# condition_value_6 = "sports"

def evaluate_single_model(model_path, model_architecture):
    if model_architecture ==  "bart":
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        tokenizer = BartTokenizer.from_pretrained(model_path)

        test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=None)
        test_df = test_df[["article", "title"]]
        test_df = test_df.dropna()
        test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 769 and len(tokenizer(x)["input_ids"]) <= 1024)]
        
    if model_architecture ==  "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=None)
        test_df = test_df[["article", "title"]]
        test_df = test_df.dropna()
        test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 769 and len(tokenizer(x)["input_ids"]) <= 1024)]

    if model_architecture ==  "bart":
        test_documents = test_df["article"].tolist() 
        test_titles = test_df["title"].tolist() 
        
    elif model_architecture == "gpt2":
        test_documents = test_df["article"].tolist() 
        test_documents = [f"######\nArticle:\n{doc}\n######\nTitle:\n" for doc in test_documents]
        test_titles = test_df["title"].tolist() 
        

    batch_size = 128

    all_pred_titles = list()
    all_gt_titles = list()

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    for i in tqdm(range(0, len(test_documents), batch_size)):
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

                if model_architecture ==  "bart":
                    inputs[j] = torch.cat([inputs[j], torch.tensor([pad_token_id] * padding_size)])
                elif model_architecture == "gpt2":
                    inputs[j] = torch.cat([torch.tensor([pad_token_id] * padding_size), inputs[j]])
                    attention_masks[j] = torch.cat([torch.tensor([0] * padding_size), attention_masks[j]])

                filtered_inputs.append(inputs[j])
                # filtered_attention_masks.append(attention_masks[j])
                filtered_gt_titles.append(test_titles[i+j])

        filtered_inputs = torch.cat([tokens.unsqueeze(0) for tokens in filtered_inputs], dim=0)
        filtered_inputs = filtered_inputs.to(device).long()

        # filtered_attention_masks = torch.cat([tokens.unsqueeze(0) for tokens in filtered_attention_masks], dim=0)
        # filtered_attention_masks = filtered_attention_masks.to(device).long()

        if model_architecture ==  "bart":
            titles = model.generate(filtered_inputs, max_length=30, num_beams=1, early_stopping=False)
        # elif model_architecture == "gpt2":
        #     titles = model.generate(filtered_inputs, attention_mask=filtered_attention_masks, num_beams=1, early_stopping=False)
        
        batch_titles = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
        # print(batch_titles[0])
        # print(batch_titles[1])
            
        if model_architecture == "gpt2":
            batch_titles = [title.split("######")[2][8:] for title in batch_titles]
            
        all_pred_titles.extend(batch_titles)
        all_gt_titles.extend(filtered_gt_titles)

    # print(len(all_pred_titles))
    # print(all_pred_titles[0])
    # print(all_gt_titles[0])
    rouge_scores = rouge_metric.compute(
        predictions=all_pred_titles,
        references=all_gt_titles,
    )

    mid_fmeasure_values = {key: value.mid.fmeasure for key, value in rouge_scores.items()}
    print(mid_fmeasure_values)

    with open(f'bartlengths_{model_path.replace("/", "_")}.pkl', 'wb') as file:
        pickle.dump([len(p.split()) for p in all_pred_titles], file)

def evaluate_double_model(summ_model_path, tg_model_path, model_architecture):
    if model_architecture ==  "bart":
        tokenizer = BartTokenizer.from_pretrained(model_path)

        test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=None)
        test_df = test_df[["article", "title"]]
        test_df = test_df.dropna()
        test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 769 and len(tokenizer(x)["input_ids"]) <= 1024)]
        
    if model_architecture ==  "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        test_df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=None)
        test_df = test_df[["article", "title"]]
        test_df = test_df.dropna()
        test_df = test_df[test_df['article'].apply(lambda x: len(tokenizer(x)["input_ids"]) >= 769 and len(tokenizer(x)["input_ids"]) <= 1024)]
        
    test_documents = test_df["article"].tolist() 
    test_titles = test_df["title"].tolist() 
    
    tokenizer = BartTokenizer.from_pretrained(tg_model_path)

    summ_model = BartForConditionalGeneration.from_pretrained(summ_model_path)
    summ_model.to(device)

    tg_model = BartForConditionalGeneration.from_pretrained(tg_model_path)
    tg_model.to(device)

    batch_size = 128

    all_pred_titles = list()
    all_gt_titles = list()

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    for i in tqdm(range(0, len(test_documents), batch_size)):
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

                if model_architecture ==  "bart":
                    inputs[j] = torch.cat([inputs[j], torch.tensor([pad_token_id] * padding_size)])
                elif model_architecture == "gpt2":
                    inputs[j] = torch.cat([torch.tensor([pad_token_id] * padding_size), inputs[j]])
                    attention_masks[j] = torch.cat([torch.tensor([0] * padding_size), attention_masks[j]])

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
        batch_titles = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
    #     titles = model.generate(filtered_inputs, max_length=30, num_beams=1, early_stopping=False)
        
    #     batch_titles = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
    #     print(batch_titles[0])
    #     print(batch_titles[1])
            
    #     if model_architecture == "gpt2":
    #         batch_titles = [title.split("######")[2][8:] for title in batch_titles]
            
        all_pred_titles.extend(batch_titles)
        all_gt_titles.extend(filtered_gt_titles)

    print(len(all_pred_titles))
    print(all_pred_titles[0])
    print(all_gt_titles[0])
    rouge_scores = rouge_metric.compute(
        predictions=all_pred_titles,
        references=all_gt_titles,
    )

    mid_fmeasure_values = {key: value.mid.fmeasure for key, value in rouge_scores.items()}
    print(mid_fmeasure_values)

    with open(f'bartlengths_{tg_model_path.replace("/", "_")}.pkl', 'wb') as file:
        pickle.dump([len(p.split()) for p in all_pred_titles], file)

        


model_path = "results/bart/doc2title/all/checkpoint-44810"
evaluate_single_model(model_path, "bart")

model_path = "results/bart/doc2title_plus/all/checkpoint-44810"
evaluate_single_model(model_path, "bart")

tg_model_path = "results/bart/summ2title/all/checkpoint-32376"
summ_model_path = "results/bart/doc2summ/checkpoint-30000"
evaluate_double_model(summ_model_path, tg_model_path, "bart")