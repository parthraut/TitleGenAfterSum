import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# input csv
train_fp = "data/atn-filtered-publication-section-train.csv"
val_fp = "data/atn-filtered-publication-section-val.csv"
test_fp = "data/atn-filtered-publication-section-test.csv"

conditions = [("all", "all"),
              ("publication", "The New York Times"), 
              ("publication", "Vice"), 
              ("publication", "TMZ"), 
              ("section", "politics"), 
              ("section", "Financials"), 
              ("section", "sports")]

chunk_size = 1e4

split_freqs = dict()
split_perc = dict()

for condition in conditions:
    col, val = condition
    split_freqs[val] = dict()
    split_perc[val] = dict()

    sizes = list()
    for csv_fp in [train_fp, val_fp, test_fp]:
        filtered_size = 0
        for chunk in pd.read_csv(csv_fp, chunksize=chunk_size, nrows=None):
            if not col == "all":
                chunk = chunk[chunk[col] == val]
            filtered_size += len(chunk)
        sizes.append(filtered_size)

    split_freqs[val]["train"] = int(sizes[0])
    split_freqs[val]["val"] = int(sizes[1])
    split_freqs[val]["test"] = int(sizes[2])

    tot = sum(sizes)
    split_perc[val]["train"] = sizes[0] / tot
    split_perc[val]["val"] = sizes[1] / tot
    split_perc[val]["test"] = sizes[2] / tot

print(split_freqs)

df = pd.DataFrame(split_freqs).astype(int)
ax = df.transpose().plot(kind="bar", stacked=True)
for c in ax.containers:
    labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center')
plt.xlabel("filter")
plt.ylabel("train-val-test frequency")
plt.xticks(rotation=90)
plt.savefig(f'figures/splits/filtered_splits_labels.png', bbox_inches='tight')

plt.clf()

df = pd.DataFrame(split_freqs).astype(int)
ax = df.transpose().plot(kind="bar", stacked=True)
plt.xlabel("filter")
plt.ylabel("train-val-test frequency")
plt.xticks(rotation=90)
plt.savefig(f'figures/splits/filtered_splits.png', bbox_inches='tight')
