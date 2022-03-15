import numpy as np
from sklearn.linear_model import LogisticRegression

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def bootstrap(df, function, n=1000, ci=95, intervention=['a', 'b', 'c', 'd'], **kwargs):
    """
    Resample the dataframe `n` times. For each resampled dataframe,
        call `function(new_df, **kwargs)` and save its value.
    Return the confidence interval that covers the central `ci`% of these values
    
    If the passed `function()` returns an array of shape [6,],
        then `bootstrap` should return an array of shape [2, 6]
        where the first row contains the bottom of the confidence interval
        and the second row contains the top of the confidence interval

    You may want to use `df.sample` and `np.percentile`
    """
    np.random.seed(42)  # Keep this random seed call for reproducibility
    results = []
    a1_vals = np.unique(df[intervention[0]])
    a2_vals = np.unique(df[intervention[1]])
    while len(results) < n:

        new_df = df.sample(frac=1, replace=True)
        result = function(new_df, intervention, a1_vals, a2_vals, **kwargs)
        # check if result is None, which means we couldn't fit all models
        if result is not None:
            results.append(result)
    results = np.array(results)
    diff = (100-ci)/2
    done = np.percentile(results, [diff, 100-diff], axis=0)
    mean = np.percentile(results, 50, axis=0)
    return done, mean

def backdoor(df, intervention=["a1", "a2", "a3", "a4"], a1_vals=[], a2_vals=[], outcome="y", confounders=["c", "d"]):
    """
    A backdoor adjustment estimator for E[Y^a]
    Use smf.ols to train an outcome model E(Y | A, confounders)

    Arguments
      df: a data frame for which to estimate the causal effect
      confounders: the variables to treat as confounders
          For the data we consider, if you include both c and d as confounders,
          this estimator should be unbiased. If you only include one,
          you would expect to see more bias.

    Returns
        results: an array of E[Y^a] estimates
    """
    big_results = [0] * len(a1_vals)
    for i, a1_val in enumerate(a1_vals):
        data = df[df[intervention[0]]==a1_val]
        results = [0] * len(a2_vals)
        for j, a2_val in enumerate(a2_vals):
            data2 = data[data[intervention[1]]==a2_val]
            if data2.empty:
                continue

            # Give up if this LogisticRegression will have only label in drafted
            if np.unique(data2["drafted"]).shape[0] == 1:
                return

            model = LogisticRegression(max_iter=1000) # Q2.3
            model.fit(data2[confounders], data2[outcome])
            results[j] = np.mean(model.predict(df[confounders]))
 
        big_results[i] = results
    return np.array(big_results)