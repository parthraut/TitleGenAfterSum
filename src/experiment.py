from datasets import load_metric
import nltk
from nltk import sent_tokenize
import evaluate

# defining metrics
bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")

predictions = ["hello there general kenobi", "foo bar foobar"]
references = ["hello there general kenobi", "hello there !"]

rouge = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
bleu = bleu_metric.compute(predictions=predictions, references=references)

print(rouge)

print({"rouge1": rouge['rouge1'], "bleu": bleu['bleu']})