import pandas as pd
from multiprocessing import Pool, cpu_count

print("---------------")
print("setting input parameters")

# input csv
input_csv_path = "data/all-the-news-2-1.csv"

# conditions
condition_column_1 = "publication"
condition_value_1 = "The New York Times"

condition_column_2 = "publication"
condition_value_2 = "Vice"

condition_column_3 = "publication"
condition_value_3 = "TMZ"

condition_column_4 = "section"
condition_value_4 = "politics"

condition_column_5 = "section"
condition_value_5 = "Financials"

condition_column_6 = "section"
condition_value_6 = "sports"

# output csv
output_csv_path = "data/atn-filtered-publication-section.csv"

chunk_size = 1e4
def filter_chunk(chunk):
    return chunk[
        (chunk[condition_column_1] == condition_value_1) |
        (chunk[condition_column_2] == condition_value_2) |
        (chunk[condition_column_3] == condition_value_3) |
        (chunk[condition_column_4] == condition_value_4) |
        (chunk[condition_column_5] == condition_value_5) |
        (chunk[condition_column_6] == condition_value_6)
    ]

print("---------------")
print("chunking")

chunks = pd.read_csv(input_csv_path, chunksize=chunk_size)
num_processes = cpu_count()
with Pool(processes=num_processes) as pool:
    filtered_dataframes = pool.map(filter_chunk, chunks)

print("---------------")
print("concatenating chunks")

filtered_data = pd.concat(filtered_dataframes, ignore_index=True)
filtered_data.to_csv(output_csv_path, index=False)