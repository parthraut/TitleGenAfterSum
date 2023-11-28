from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"

print("-----------------------")
print("Setting model and tokenizer")

# model_dir = "results/bart/doc2title_plus/all/checkpoint-10755 "
# # model_dir = "results/bart/doc2title_plus/tmz/checkpoint-1500"
# model = BartForConditionalGeneration.from_pretrained(model_dir)
# model.to(device)
# tokenizer = BartTokenizer.from_pretrained(model_dir)

# Define your input text
nrows = 50
df = pd.read_csv("./data/atn-filtered-publication-section-train-bart-summ.csv", nrows=nrows)

for article_num in range(15, nrows):
    input_text = df["article"][article_num]
    print("Input Article:\n", input_text)

    # # Tokenize the input text
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    # input_ids = input_ids[:, :512]
    # input_ids = input_ids.to(device)

    # # Generate output
    # output_ids = model.generate(input_ids, max_length=200)

    # # Decode the output
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # # Print the generated output
    # print("Generated Title: ", output_text)
    # print("GT Title:", df["title"][article_num])

    input("Click for the next article...")