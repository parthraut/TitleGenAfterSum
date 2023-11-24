import pandas as pd
import numpy as np

original_df = pd.read_csv('data/atn-filtered-publication-section.csv')
np.random.seed(42)

indices = np.arange(len(original_df))
np.random.shuffle(indices)

total_size = len(original_df)
split1_size = int(0.7 * total_size)
split2_size = int(0.15 * total_size)

split1_indices = indices[:split1_size]
split2_indices = indices[split1_size:(split1_size + split2_size)]
split3_indices = indices[(split1_size + split2_size):]

split1_df = original_df.iloc[split1_indices]
split2_df = original_df.iloc[split2_indices]
split3_df = original_df.iloc[split3_indices]

split1_df.to_csv('data/atn-filtered-publication-section-train.csv', index=False)
split2_df.to_csv('data/atn-filtered-publication-section-val.csv', index=False)
split3_df.to_csv('data/atn-filtered-publication-section-test.csv', index=False)