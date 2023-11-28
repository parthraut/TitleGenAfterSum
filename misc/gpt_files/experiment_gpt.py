from transformers import AutoConfig, GPT2LMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"

print("-----------------------")
print("Setting model and tokenizer")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=1024,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
model.to(device)

# Define your input text
nrows = 50
df = pd.read_csv("./data/atn-filtered-publication-section-val.csv", nrows=nrows)

for article_num in range(15, nrows):
    input_text = df["article"][article_num]
    input_text = f"######\nArticle:\n{input_text}\n######\nTitle:\n"
    print(input_text)

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids[:, :512]
    input_ids = input_ids.to(device)

    # Generate output
    output_ids = model.generate(input_ids, max_length=200)

    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print the generated output
    print("Output: \n", output_text)
    print("GT Title:", df["title"][article_num])

    input("Click for the next article...")