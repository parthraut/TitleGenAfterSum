from transformers import GPT2LMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"

print("-----------------------")
print("Setting model and tokenizer")

model_dir = "results/bart/doc2title_plus/all/checkpoint-44810"
# model_dir = "results/bart/doc2title_plus/tmz/checkpoint-1500"
model = BartForConditionalGeneration.from_pretrained(model_dir)
model.to(device)
tokenizer = BartTokenizer.from_pretrained(model_dir)

# model = GPT2LMHeadModel.from_pretrained("results/gpt2/doc2title/all/checkpoint-17924")
# model.to(device)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token

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
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    output_ids = model.generate(input_ids, max_length=1024, attention_mask=attention_mask, repetition_penalty=2.0, num_return_sequences=1)

    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print the generated output
    print("Generated Title: ")
    print(output_text)
    print("GT Title:", titles[article_num])

    input("Click for the next article...")