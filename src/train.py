import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from dataset_preparation import load_hf_data, load_csv_data
import torch
import nltk
from nltk import sent_tokenize

# downloading nltk
nltk.download('punkt')

# defining metrics
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

# function to compute metrics
def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds

    preds[preds == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    preds_tokens_list = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]

    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels_tokens_list = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    bleu = bleu_metric.compute(predictions=preds_tokens_list, references=labels_tokens_list)
    rouge = rouge_metric.compute(predictions=preds_tokens_list, references=labels_tokens_list)

    return {"bleu": bleu['bleu'], "rouge1": rouge['rouge1'], "rouge2": rouge['rouge2'], "rougeL": rouge['rougeL']}

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("-----------------------")
print("Setting model and tokenizer")

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

print("-----------------------")
print("Loading data")

train_ds, val_ds, test_ds = load_csv_data(tokenizer, "../data/all-the-news-2-1.csv", nrows=1e4, return_dl=False)

print("-----------------------")
print("Setting training arguments")

# Set up Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
compute_metrics_lambda = lambda x: compute_metrics(tokenizer, x)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics_lambda,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("-----------------------")
print("Training the model")

# Train your model
trainer.train()
