import pandas as pd
import estimate_and_graph

df = pd.read_csv('archive/college_and_combine.csv')

estimate_and_graph.estimate_and_graph(df.copy(), type='drafted', img_path='images/drafted.png')
estimate_and_graph.estimate_and_graph(df.copy(), type='rounds', img_path='images/rounds.png')
estimate_and_graph.estimate_and_graph(df.copy(), type='picks', img_path='images/picks.png')
estimate_and_graph.estimate_and_graph(df.copy(), type='only picked', img_path='images/only_picked.png')


df.drop(columns=['lane', 'vert', 'bench', 'weight'], inplace=True)
mice_df = pd.read_csv('archive/our_testing.csv')
mice_df = df.merge(right=mice_df, on='Unnamed: 0')

estimate_and_graph.estimate_and_graph(mice_df.copy(), type='drafted', img_path='images/mice_drafted.png')
estimate_and_graph.estimate_and_graph(mice_df.copy(), type='drafted', img_path='images/mice_drafted_bench.png',
    interventions=['bench', 'vert'], labels=['Max Bench Press (reps)', 'Standing Vertical Leap (in)'])
