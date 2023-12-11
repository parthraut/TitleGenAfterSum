from transformers import GPT2LMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"

print("-----------------------")
print("Setting model and tokenizer")

summ_model_dir = "results/bart/doc2summ/checkpoint-30000"
summ_model = BartForConditionalGeneration.from_pretrained(summ_model_dir)
summ_model.to(device)

tgmodel_dir = "results/bart/summ2title/all/checkpoint-32376"
tg_model = BartForConditionalGeneration.from_pretrained(tgmodel_dir)
tg_model.to(device)

tokenizer = BartTokenizer.from_pretrained(tgmodel_dir)

# Define your input text
nrows = 500
df = pd.read_csv("./data/atn-filtered-publication-section-test.csv", nrows=nrows)
df = df[["article", "title"]]
df = df.dropna()

articles = list(df["article"])
titles = list(df["title"])

for article_num in range(15, nrows):
    input_text = articles[article_num]
    # input_text = f"######\nArticle:\n{input_text}\n######\nTitle:\n"
    # print("Input Article:\n", input_text)

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    if len(input_ids[0]) > 1024:
        continue 
    
    input_ids = input_ids[:, :1024]
    input_ids = input_ids.to(device)

    # Generate output
    # attention_mask = (input_ids != tokenizer.pad_token_id).long()
    output_ids = summ_model.generate(input_ids, max_length=1024, repetition_penalty=2.0, num_return_sequences=1)
    output_ids = tg_model.generate(output_ids, max_length=1024, repetition_penalty=2.0, num_return_sequences=1)

    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print the generated output
    print("Generated Title: ")
    print(output_text)
    print("GT Title:", titles[article_num])

    input("Click for the next article...")