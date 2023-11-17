import pandas as pd
from multiprocessing import Pool
from matplotlib import pyplot as plt

chunk_size = 1e4
num_processes = 70
nrows = None

def process_chunk(chunk):
    return {
        "publication": chunk["publication"].value_counts(),
        "section": chunk["section"].value_counts()
    }

def get_freq(value_counts):
    features = list(value_counts[0].keys())
    freqs = {key: {} for key in features}

    for result in value_counts:
        for feature in freqs:
            for k, v in result[feature].items():
                freqs[feature][k] = freqs[feature].get(k, 0) + v

    return freqs

def plot(freqs):
    for key in freqs:
        dictionary = freqs[key]
        dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True)[:10])

        categories = list(dictionary.keys())
        values = list(dictionary.values())

        plt.bar(categories, values)
        plt.xlabel(key)
        plt.ylabel('frequency')
        plt.title(f'frequency of {key}')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(f'./figures/freq_{key}.png')
        plt.show()
        plt.clf()

pool = Pool(processes=num_processes)

chunk_list = []
for chunk in pd.read_csv("./data/all-the-news-2-1.csv", chunksize=chunk_size, nrows=nrows):
    chunk.dropna()
    chunk_list.append(chunk)

publication_value_counts = pool.map(process_chunk, chunk_list)
freqs = get_freq(publication_value_counts)

plot(freqs)