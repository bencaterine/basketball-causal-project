import pandas as pd
import numpy as np
import os
import estimator
import matplotlib.pyplot as plt

N_bins = 4

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
# df['first'] = (df['pick'] < 31)*2
# df['second'] = (df['pick'] > 30)*1
# df['drafted'] = df['first'] + df['second']
df['drafted'] = df['pick'].fillna(61)
print(df[['pick', 'drafted']])

# df['vert_avg'] = df[['vert', 'mvert']].mean(axis=1)
# df['run_avg'] = df[['lane', 'sprint']].mean(axis=1)

df['lane'] = pd.qcut(df['lane'], N_bins)
df['sprint'] = pd.qcut(df['sprint'], N_bins)
df['vert'] = pd.qcut(df['vert'], N_bins)
df['mvert'] = pd.qcut(df['mvert'], N_bins)
# df['vert_avg'] = pd.qcut(df['vert_avg'], N_bins)
# df['run_avg'] = pd.qcut(df['run_avg'], N_bins)

# specific stats we'll be using for analysis
combine_stats_to_use = [
    'fat', 'hand_length', 'hand_width', 'height', 'reach', 'weight', 'WINGSPAN'
]
college_stats_to_use = [
    'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'TO_per', 'FT_per', 'twoP_per', 'TP_per', 'ftr',
    'porpag', 'adjoe', 'atr', 'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm',
    'gbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts'
]
intervention = ['lane', 'vert'] # ['run_avg', 'vert_avg']

# narrow the df before dropping null values
specific_df = df[['drafted'] + intervention + combine_stats_to_use + college_stats_to_use]
specific_df = specific_df.dropna()
print(specific_df)

# perform backdoor using bootstrap to get CIs
est, mean = estimator.bootstrap(
    specific_df,
    function=estimator.backdoor,
    n=100,
    intervention=intervention,
    outcome='drafted',
    confounders = combine_stats_to_use + college_stats_to_use
)

# visualize resulting CI matrices
i0_ticks = [np.around(np.unique(specific_df[intervention[0]])[k].left, decimals=3) for k in [0,1,2,3]]
i0_ticks.append(np.around(np.unique(specific_df[intervention[0]])[3].right, decimals=3))
i1_ticks = [np.around(np.unique(specific_df[intervention[1]])[k].left, decimals=3) for k in [0,1,2,3]]
i1_ticks.append(np.around(np.unique(specific_df[intervention[1]])[3].right, decimals=3))
rounded = np.append(np.around(est, decimals=3), [np.around(mean, decimals=3)], axis=0)
print(rounded)
fig, axs = plt.subplots(1,3)
fig.set_size_inches(11, 3)
titles = ['Lower CI', 'Upper CI', 'Mean']
for p in [0,1,2]:
    caxes = axs[p].matshow(rounded[p], interpolation='nearest')
    fig.colorbar(caxes, ax=axs[p])
    for i in range(N_bins):
        for j in range(N_bins):
            c = rounded[p,j,i]
            axs[p].text(i, j, str(c), va='center', ha='center')
    axs[p].set_title(titles[p])
    axs[p].set_xticks([-.5,.5,1.5,2.5,3.5], i0_ticks)
    axs[p].set_yticks([-.5,.5,1.5,2.5,3.5], i1_ticks)
    axs[p].set_xlabel('Lane Agility Time (s)')
    axs[p].xaxis.set_label_position('top')
    axs[p].set_ylabel('Standing Vertical Leap (in)')
plt.tight_layout()
plt.savefig('ci_matrices_picks_new.png', dpi=500)