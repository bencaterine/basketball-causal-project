import pandas as pd
import os

dir = os.path.dirname(__file__)
stats = os.path.join(dir, 'archive/stats09-21.csv')

stats_cols = {
    'points': 'pts',
    'assists': 'ast',
    'rebounds': 'treb',
    'blocks': 'blk_per',
    'shooting pct': 'TS_per',
    'recruiting rank': 'Rec Rank'
}

df = pd.read_csv(stats)
print(df)

for stat in stats_cols.keys():
    print(stat, 'mean:', df[stats_cols[stat]].mean())
    print(stat, 'std:', df[stats_cols[stat]].std())