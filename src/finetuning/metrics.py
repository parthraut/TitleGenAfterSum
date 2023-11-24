import evaluate
from nltk import sent_tokenize
import nltk

# downloading nltk
nltk.download('punkt')

# defining metrics
rouge_metric = evaluate.load("rouge")

# function to compute metrics
def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds

    preds[preds == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    preds_tokens_list = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]

    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels_tokens_list = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    rouge = rouge_metric.compute(predictions=preds_tokens_list, references=labels_tokens_list)

    return {"rouge1": rouge['rouge1'], "rouge2": rouge['rouge2'], "rougeL": rouge['rougeL']}