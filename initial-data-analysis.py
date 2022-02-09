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

# df = pd.read_csv(stats)
# print('points mean:', df['pts'].mean())
# print('points std:', df['pts'].std())

# #assists
# df = pd.read_csv(stats)
# print('points mean:', df['ast'].mean())
# print('points std:', df['ast'].std())

# #total rebounds
# df = pd.read_csv(stats)
# print('points mean:', df['treb'].mean())
# print('points std:', df['treb'].std())

# #blocks
# df = pd.read_csv(stats)
# print('points mean:', df['blk_per'].mean())
# print('points std:', df['blk_per'].std())

# #true shooting %
# df = pd.read_csv(stats)
# print('points mean:', df['TS_per'].mean())
# print('points std:', df['TS_per'].std())

# #recruiting rank
# print('points mean:', df['Rec rank'].mean())
# print('points std:', df['Rec rank'].std())