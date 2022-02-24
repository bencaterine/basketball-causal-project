import numpy as np
import statsmodels.formula.api as smf
import pandas as pd


def naive(df):
    """
    Naive estimator that assumes E[Y^a] = E[Y | A]
    Should work when A is randomized
    """
    results = []
    for i in sorted(np.unique(df["a"])):
        est = np.mean(df[df["a"] == i]["y"])
        results.append(est)
    return np.array(results)


def naive_linear(df):
    """
    A naive linear estimator for E[Y^a]
    Assumes E[Y^a] = E[Y | A]
    Should work when A is randomized and A->Y is linear
    """
    params = smf.ols("y~a", data=df).fit().params.to_numpy()
    n_a = np.unique(df["a"]).shape[0]
    return np.stack([np.ones(n_a), np.arange(n_a)], axis=1).dot(params)


def bootstrap(df, function, n=1000, ci=95, **kwargs):
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
    for _ in range(n):
        new_df = df.sample(frac=1, replace=True)
        results.append(function(new_df, **kwargs))
    results = np.array(results)
    diff = (100-ci)/2
    cis = []
    for col in results.T:
        cis.append([np.percentile(col, diff), np.percentile(col, 100-diff)])
    done = np.array(cis)
    return done.transpose()

def backdoor(df, intervention="a", outcome="y", confounders=["c", "d"]):
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
    expression = outcome+'~'+'+'.join(confounders)
    a_vals = np.unique(df[intervention])
    mean_df = df.mean()
    results = []
    for a_val in a_vals:
        params = smf.ols(expression, data=df[df[intervention]==a_val]).fit().params
        results.append(params['Intercept'] + sum([params[conf] * mean_df[conf] for conf in confounders]))
    return np.array(results)


def ipw(df, confounders=["c", "d"]):
    """
    An inverse probability weighting estimator for E[Y^a]
    You may want to use smf.mnlogit to train a propensity model p(A | confounders)

    Arguments
      df: a data frame for which to estimate the causal effect
      confounders: the variables to treat as confounders.
          For the data we consider, if you include both c and d as confounders,
          this estimator should be unbiased. If you only include one,
          you would expect to see more bias.

    Returns
        results: an array of E[Y^a] estimates
    """
    model = smf.mnlogit('a~'+'+'.join(confounders), data=df).fit()
    a_vals = np.unique(df['a'])
    predictions = model.predict(df[confounders])
    frames = [df, predictions]
    df = pd.concat(frames, axis=1)
    df['pred'] = 0
    df.index = range(df.shape[0])
    for a_val in a_vals:
        df.loc[df['a']==a_val, 'pred'] = df[a_val]
    df['partial'] = 1/df.shape[0] * df['y'] * 1/df['pred']
    df = df.drop([a_val for a_val in a_vals], axis=1)
    results = df.groupby(['a']).sum()['partial'].values
    return np.array(results)


def frontdoor(df):
    """
    A front-door estimator for E[Y^a]
    Should only use a, m, and y -- not c or d.
    You may want to use smf.ols to model E[Y | M, A]

    Arguments
      df: a data frame for which to estimate the causal effect

    Returns
        results: an array of E[Y^a] estimates
    
    """
    results = []
    raise NotImplementedError

    return np.array(results)
