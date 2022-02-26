import pandas as pd
import numpy as np
import os
import estimator


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
college_df.drop_duplicates('player_name', keep='last', inplace=True)
# print(college_df)
# print(college_df.info())

# for stat in stats_cols.keys():
#     print(stat, 'mean:', college_df[stats_cols[stat]].mean())
#     print(stat, 'std:', college_df[stats_cols[stat]].std())

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
combine_df.drop_duplicates('PLAYER', keep='last', inplace=True)
# merge college and combine data
df = combine_df.merge(right=college_df, left_on='PLAYER', right_on='player_name')

# transform data
df.rename(
    {
        'BODY_FAT_%': 'fat',
        'HAND_LENGTH_(INCHES)': 'hand_length',
        'HAND_WIDTH_(INCHES)': 'hand_width',
        'HEIGHT_W/O_SHOES': 'height',
        'STANDING_REACH': 'reach',
        'WEIGHT_(LBS)': 'weight',
        'LANE_AGILITY_TIME_(SECONDS)': 'lane',
        'THREE_QUARTER_SPRINT_(SECONDS)': 'sprint',
        'STANDING_VERTICAL_LEAP_(INCHES)': 'vert',
        'MAX_VERTICAL_LEAP_(INCHES)': 'mvert',
        'ast/tov': 'atr'
    },
    axis=1,
    inplace=True
)
df['drafted'] = ~df['pick'].isnull()*1
df['lane'] = pd.qcut(df['lane'], 10)
df['sprint'] = pd.qcut(df['sprint'], 10)
df['vert'] = pd.qcut(df['vert'], 10)
df['mvert'] = pd.qcut(df['mvert'], 10)
combine_stats_to_use = [
    'fat', 'hand_length', 'hand_width', 'height', 'reach', 'weight', 'WINGSPAN'
]
college_stats_to_use = [
    'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'TO_per', 'FT_per', 'twoP_per', 'TP_per', 'ftr',
    'porpag', 'adjoe', 'atr', 'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm',
    'gbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts'
]
specific_df = df[['drafted', 'lane', 'vert'] + combine_stats_to_use + college_stats_to_use]
specific_df.dropna(inplace=True)
print(specific_df)

est = estimator.bootstrap(
    specific_df,
    function=estimator.backdoor,
    n=100,
    intervention=['lane', 'vert'],
    outcome='drafted',
    confounders = combine_stats_to_use + college_stats_to_use
)
print(est)

# TODO:
# - interpret CI matrix
# - nonbinary drafted outcome?