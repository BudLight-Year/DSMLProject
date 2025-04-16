import pandas as pd

sample_df = pd.read_csv("news-docs.2007.en.filtered", sep="\t", nrows=5)
print("Column names:", sample_df.columns.tolist())
print("\nSample data:")
print(sample_df.head())
