from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
import torch
import pandas as pd

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"

print("-----------------------")
print("Setting model and tokenizer")

model_dir = "results/title_generation/param_efficient/publication/Vice/checkpoint-2210"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Define your input text
df = pd.read_csv("./data/all-the-news-2-1.csv", nrows=10)
article_num = 5
input_text = df["article"][article_num]
print("Input Text:", input_text)
print("GT Title:", df["title"][article_num])

# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
input_ids = input_ids[:, :512]
input_ids = input_ids.to(device)

# Generate output
output_ids = model.generate(input_ids)

# Decode the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated output
print("Generated Output:", output_text)