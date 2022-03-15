import pandas as pd
import numpy as np
import estimator
import matplotlib.pyplot as plt

def estimate_and_graph(
    df, # dataframe to use
    type, # type of outcome variable (binary drafted, by draft pick, etc.)
    img_path, # where to save results image
    N_bins=4, # number of bins for result matrices
    N_iterations=100, # number of iterations for bootstrap
    interventions=['lane', 'vert'], # which variables to intervene on
    labels=['Lane Agility Time (s)', 'Standing Vertical Leap (in)']): # matrix labels for these variables

    print('Started', img_path, '...')

    # Q3.1 Multiple draft outcome variable formats
    # fill in drafted column based on which outcome variable type is being used
    if type=='drafted':
        # binary (drafted or not)
        df['drafted'] = ~df['pick'].isnull()*1
    elif type=='rounds':
        # by round (1st, 2nd, undrafted)
        df['first'] = (df['pick'] < 31)*2
        df['second'] = (df['pick'] > 30)*1
        df['drafted'] = df['first'] + df['second']
    elif type=='picks':
        # by pick number 1-60, undrafted=61
        df['drafted'] = df['pick'].fillna(61)
    elif type=='only picked':
        # by pick number, only drafted players (1-60)
        df['drafted'] = df['pick']
        df.dropna(subset=['drafted'], inplace=True)
    
    # bin the combine drill variables
    df['lane'] = pd.qcut(df['lane'], N_bins)
    df['sprint'] = pd.qcut(df['sprint'], N_bins)
    df['vert'] = pd.qcut(df['vert'], N_bins)
    df['mvert'] = pd.qcut(df['mvert'], N_bins)
    df['bench'] = pd.qcut(df['bench'], N_bins)

    # specific stats we'll be using for analysis
    combine_stats_to_use = [
        'fat', 'hand_length', 'hand_width', 'height', 'reach', 'weight', 'WINGSPAN'
    ]
    college_stats_to_use = [
        'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'TO_per', 'FT_per', 'twoP_per', 'TP_per', 'ftr',
        'porpag', 'adjoe', 'atr', 'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm',
        'gbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts'
    ]
    intervention = interventions

    # narrow the df before dropping null values
    specific_df = df[['drafted'] + intervention + combine_stats_to_use + college_stats_to_use]
    specific_df = specific_df.dropna()
    print(specific_df)

    # perform backdoor using bootstrap to get CIs
    est, mean = estimator.bootstrap(
        specific_df,
        function=estimator.backdoor,
        n=N_iterations,
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
    # Q3.1 Added 'mean' matrix in addition to confidence interval matrices
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
        axs[p].set_xlabel(labels[0])
        axs[p].xaxis.set_label_position('top')
        axs[p].set_ylabel(labels[1])
    plt.tight_layout()
    # save to img_path
    plt.savefig(img_path, dpi=500)

    print('Finished', img_path)