import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# input csv
train_fp = "data/atn-filtered-publication-section-train.csv"
val_fp = "data/atn-filtered-publication-section-val.csv"
test_fp = "data/atn-filtered-publication-section-test.csv"

# lengths = {
#     "train": list(),
#     "val": list(),
#     "test": list()
# }

print("loading train")
train_df = pd.read_csv(train_fp)
train_df = train_df.dropna()
train_lengths = [len(article.split()) for article in train_df["article"] if len(article.split()) <= 1024]
train_lengths = [i for i in train_lengths if i <= 4000]
# lengths["train"] = train_lengths

print("loading val")
val_df = pd.read_csv(val_fp)
val_df = val_df.dropna()
val_lengths = [len(article.split()) for article in val_df["article"] if len(article.split()) <= 1024]
val_lengths = [i for i in val_lengths if i <= 4000]
# lengths["val"] = val_lengths

print("loading test")
test_df = pd.read_csv(test_fp)
test_df = test_df.dropna()
test_lengths = [len(article.split()) for article in test_df["article"] if len(article.split()) <= 1024]
test_lengths = [i for i in test_lengths if i <= 4000]
# lengths["test"] = test_lengths

n_bins = 50

plt.hist(train_lengths, bins=n_bins, alpha=0.5, label='train')
plt.hist(val_lengths, bins=n_bins, alpha=0.5, label='val')
plt.hist(test_lengths, bins=n_bins, alpha=0.5, label='test')
plt.legend(loc='upper right')
plt.xlabel("number of words")
plt.ylabel('frequency')
plt.title(f'article word count')
plt.savefig(f'./figures/length/article_length.png')