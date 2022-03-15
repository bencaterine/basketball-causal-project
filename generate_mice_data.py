#Q3.1
# # Analyzing missing data in Python Statsmodels

# This is taken from a MICE Jupiter notebook that demonstrates several techniques
# for working with missing data in Python, using the Statsmodels library.

# First we import the libraries that we will be using.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.imputation import mice
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI

# Here we load the data from the college_and_combine into a dataframe df

df = pd.read_csv('archive/college_and_combine.csv')

# # Multiple imputation

# Here we demonstrate how to use multiple imputation to estimate a
# correlation coefficient when some data values are missing.  This is first
# just practice using the MICE jupiter notebook to get familirar before
# actually imputing missing values

# Here we count the number of missing values

dx = df.loc[:, ["sprint", "bench"]]

# Size of all data, including missing values
print(dx.shape)

# Number of missing values for each variable
print(pd.isnull(dx).sum(0))

# Number of cases that are missing both variables
print(pd.isnull(dx).prod(1).sum(0))


# Next, for comparison purposes, we estimate the correlation
# coefficient and its standard error using "complete case" analysis:

dd = dx.dropna()
c = np.cov(dd.T)

r_cc = c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
print("Complete case estimate:       %f" % r_cc)
print("Complete case standard error: %f\n" % (1 / np.sqrt(dd.shape[0])))
# -

bm = BayesGaussMI(dx, mean_prior=100*np.eye(2), cov_prior=100*np.eye(2))

for k in range(100):
    bm.update()

# Now we are ready to draw samples from the imputation object, and use
# these samples to estimate the unknown parameter of interest

rv = []
for k in range(200):
    bm.update()
    # After calling bm.update, we can access bm.mean and bm.cov,
    # which are draws from the posterior distribution of the
    # Gaussian mean and covariance parameters given the data.
    r = bm.cov[0, 1] / np.sqrt(bm.cov[0, 0] * bm.cov[1, 1])

    rv.append(r)

rv = np.asarray(rv)

# Based on these posterior samples, we can estimate the posterior mean and
# posterior variance of the correlation

print("Mean: ", rv.mean())
print("SD:   ", rv.std())


# # Here we begin actually using MICE on our data.

# Description as provided by the notebook:
# Multiple Imputation with Chained Equations (MICE) is a
# regression-based framework for imputing missing values that allows
# us to specify arbitrary regression models for imputing each
# variable's missing values from the other variables.

# One common workflow with MICE is to create a set of imputed
# datasets, then save them as files.  They can then be retrieved later
# and used in an MI analysis using the "combining rules".  This
# workflow is illustrated below.

# Take the columns with missing values

dx = df.copy()
dx = dx.loc[:, ["lane", "weight", "vert", "bench"]]

#Implement the MICEData
imp_data = mice.MICEData(dx)

# Create a new sample of the data from MICE and output it as df2
df2 = imp_data.next_sample()

print(df2)

#Save this MICE data table to a CSV
df2.to_csv('archive/our_testing.csv')
