from dataset_preparation_gpt import load_csv_data
from transformers import AutoConfig, GPT2LMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("-----------------------")
print("Loading datasets")
train_ds, val_ds, _ = load_csv_data(
    tokenizer, 
    ("./data/atn-filtered-publication-section-train.csv", "./data/atn-filtered-publication-section-val.csv", "./data/atn-filtered-publication-section-test.csv"),
    chunk_size=10, 
    nrows=100, 
    filters=None, 
    return_dl=False,
    batch_size=32
)

lengths = [len(i["input_ids"]) for i in train_ds]
print(max(lengths))