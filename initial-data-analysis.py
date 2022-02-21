import pandas as pd
import numpy as np
import os

dir = os.path.dirname(__file__)
college_stats = os.path.join(dir, 'archive/stats09-21.csv')
combine_stats = os.path.join(dir, 'archive/nba_combine.csv')

stats_cols = {
    'points': 'pts',
    'assists': 'ast',
    'rebounds': 'treb',
    'blocks': 'blk_per',
    'shooting pct': 'TS_per',
    'recruiting rank': 'Rec Rank'
}

# college stats data
college_df = pd.read_csv(college_stats)
print(college_df)
print(college_df.info())

for stat in stats_cols.keys():
    print(stat, 'mean:', college_df[stats_cols[stat]].mean())
    print(stat, 'std:', college_df[stats_cols[stat]].std())

# combine data
combine_df = pd.read_csv(combine_stats)

# clean combine data
# convert strings
combine_df['BODY_FAT_%'] = combine_df['BODY_FAT_%'].str.slice(stop=-1)
combine_df.replace('-', np.nan, inplace=True)
# convert ft/in to inches
feet_inches = ['HEIGHT_W/O_SHOES', 'HEIGHT_W/_SHOES']
for col in ['HEIGHT_W/O_SHOES', 'HEIGHT_W/_SHOES', 'STANDING_REACH', 'WINGSPAN']:
    combine_df[col] = 12*combine_df[col].str.split("'").str[0].astype(float) + combine_df[col].str.split("'").str[1].astype(float)
# fix data types
to_float = combine_df.columns.difference(['YEAR', 'PLAYER', 'POS'])
combine_df[to_float] = combine_df[to_float].astype(float)
# drop rows with no data (keep rows with some data)
combine_df.dropna(thresh=4, inplace=True)