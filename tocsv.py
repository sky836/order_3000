import pandas as pd

df = pd.read_csv('clean_data.csv')
df = df.iloc[22468:, 2:]
df.to_csv('continues_clean_data.csv')

