from datasets import load_metric
import pandas as pd
import os

# Load the ROUGE metric
rouge_metric = load_metric("rouge")

df = pd.read_csv("./data/atn-filtered-publication-section-test2.csv")
df = df.dropna()
print(df.columns)


def evaluate(directory_path):
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    gt = df["title"].tolist()
    generated = df[model_path].tolist()

    # Calculate ROUGE scores
    rouge_scores = rouge_metric.compute(
        predictions=generated,
        references=gt,
    )

    # Print the ROUGE scores
    print(rouge_scores)



# evaluate("./results/bart/doc2title/all")
# evaluate("./results/bart/doc2title_plus/all")
evaluate("./results/bart/summ2title/all")