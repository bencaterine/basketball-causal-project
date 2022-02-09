import pandas as pd
import os

dir = os.path.dirname(__file__)
stats = os.path.join(dir, 'archive/stats09-21.csv')

df = pd.read_csv(stats)
print(df)