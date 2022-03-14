CS 396-4 Causal Inference Final Project
Ben Caterine and Michael Diersen

Run these files in order to reproduce our results:
1. initial-data-analysis.py
    This merges the college basketball player data and NBA Combine player data
    into one dataset. It also performs data cleaning to unify datatypes and
    simplify column names.
2. generate_mice_data.py
    This uses MICE to impute missing data in our dataset, such as missing
    values in the 'standing vertical' column and the entire 'bench press'
    column.
3. main.py
    This runs the actual estimator on our dataset. Specifically, it uses
    different 'drafted' outcomes (binary, by round, by pick, etc.) to produce
    differing results, and it also tests what happens if we run the estimator
    on our MICE data.
    Specifically, main.py runs estimate_and_graph.py, which runs estimator.py.
    estimator.py estimates counterfactuals for causal effect and computes
    confidence intervals, and estimate_and_graph graphs these results in 2D
    heat maps. This process is repeated for each drafted/MICE combination.