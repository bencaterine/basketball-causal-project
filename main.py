import pandas as pd
from estimate_and_graph import estimate_and_graph

# data management and cleaning
import initial_data_analysis

# impute missing values (in bench press, vertical jump, etc.)
import generate_mice_data


# non-imputed dataset (still has missing values) from initial_data_analysis
df = pd.read_csv('archive/college_and_combine.csv')


# binary drafted outcome (1=drafted, 0=not)
estimate_and_graph(df.copy(), type='drafted', img_path='images/drafted.png')

# by round (2=first round, 1=second round, 0=undrafted)
estimate_and_graph(df.copy(), type='rounds', img_path='images/rounds.png')

# by pick (1=first pick, ..., 60=sixtieth pick, 61=undrafted)
estimate_and_graph(df.copy(), type='picks', img_path='images/picks.png')

# only drafted players (1=first pick, ..., 60=sixtieth pick)
estimate_and_graph(df.copy(), type='only picked', img_path='images/only_picked.png')



# add in imputed values from generate_mice_data
df.drop(columns=['lane', 'vert', 'bench', 'weight'], inplace=True)
mice_df = pd.read_csv('archive/our_testing.csv')
mice_df = df.merge(right=mice_df, on='Unnamed: 0')


# binary drafted outcome with imputed data
estimate_and_graph(mice_df.copy(), type='drafted', img_path='images/mice_drafted.png')

# binary with new imputed bench press column
estimate_and_graph(mice_df.copy(), type='drafted', img_path='images/mice_drafted_bench.png',
    interventions=['bench', 'vert'], labels=['Max Bench Press (reps)', 'Standing Vertical Leap (in)'])
