import pandas as pd
import numpy as np

# college stats data
college_df = pd.read_csv('archive/stats09-21.csv')
college_df.drop_duplicates('player_name', keep='last', inplace=True)

# combine data
combine_df = pd.read_csv('archive/nba_combine.csv')

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
combine_df.dropna(subset='WEIGHT_(LBS)', inplace=True)
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
        'MAX_BENCH_PRESS_(REPETITIONS)': 'bench',
        'ast/tov': 'atr'
    },
    axis=1,
    inplace=True
)

df.to_csv('archive/college_and_combine.csv')