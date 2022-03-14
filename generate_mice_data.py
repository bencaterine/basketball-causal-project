# # Analyzing missing data in Python Statsmodels, a case study with the NHANES data

# This notebook demonstrates several techniques for working with
# missing data in Python, using the Statsmodels library.  The methods
# are illustrated using data from the
# NHANES (National Health and Nutrition Examination Study).

# First we import the libraries that we will be using.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.imputation import mice
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI

# Next we will load the data.  The NHANES study encompasses multiple
# waves of data collection.  Here we will only use the 2015-2016 data.

# dir = os.path.dirname(__file__)
# file_path = os.path.join(dir,'combine.csv')
df = pd.read_csv('archive/college_and_combine.csv')


# Retain a subset of columns for use below.
# vars = ["BPXSY1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "BMXBMI", "SMQ020"]
# -

# # Multiple imputation

# Here we demonstrate how to use multiple imputation to estimate a
# correlation coefficient when some data values are missing.  Blood
# pressure and BMI are expected to be positively related, and we
# estimate the correlation between them below.  A thorough understanding
# of the relationship between blood pressure and BMI should consider
# gender, BMI, and other possibly relevant factors.  But for illustration,
# we focus here on the simple unadjusted correlation.
#
# In the next cell, we determine how many values of these variables are missing:


dx = df.loc[:, ["sprint", "bench"]]
# dx = da.loc[:, ["BPXSY1", "BMXBMI"]]


# Size of all data, including missing values
print(dx.shape)

# Number of missing values for each variable
print(pd.isnull(dx).sum(0))

# Number of cases that are missing both variables
print(pd.isnull(dx).prod(1).sum(0))


# Next, for comparison purposes, we estimate the correlation
# coefficient and its standard error using "complete case" analysis:

# +
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
# these samples to estimate the unknown parameter of interest (the correlation
# between blood pressure and BMI).

# +
rv = []
for k in range(200):
    bm.update()

    # After calling bm.update, we can access bm.mean and bm.cov,
    # which are draws from the posterior distribution of the
    # Gaussian mean and covariance parameters given the data.
    # We can also access the underlying data frame dx, which
    # has now been imputed so that there are no missing values.
    r = bm.cov[0, 1] / np.sqrt(bm.cov[0, 0] * bm.cov[1, 1])

    rv.append(r)

rv = np.asarray(rv)
# -

# Based on these posterior samples, we can estimate the posterior mean and
# posterior variance of the correlation coefficient between BMI and
# blood pressure.

print("Mean: ", rv.mean())
print("SD:   ", rv.std())

# We can also view the histogram of the draws from the posterior
# distribution.

test = plt.hist(rv, bins=15, alpha=0.5)
print(test)

# # MICE

# Multiple Imputation with Chained Equations (MICE) is a
# regression-based framework for imputing missing values that allows
# us to specify arbitrary regression models for imputing each
# variable's missing values from the other variables.

# One common workflow with MICE is to create a set of imputed
# datasets, then save them as files.  They can then be retrieved later
# and used in an MI analysis using the "combining rules".  This
# workflow is illustrated below.

# +
dx = df.copy()
dx = dx.loc[:, ["lane", "weight", "vert", "bench"]]

# Recode to 0 (male), 1 (female)
# dx.RIAGENDR -= 1

# # Introduce some missing values
# for k in range(dx.shape[1]):
#     ii = np.flatnonzero(np.random.uniform(size=dx.shape[0]) < 0.1)
#     dx.iloc[ii, k] = np.nan

imp_data = mice.MICEData(dx)
# imp_data.set_imputer("Var", "Var1 + Var2")
# imp_data.set_imputer("Var", "Var1 + Var2 + Var3", model_class=sm.GLM,
#                      init_kwds={"family": sm.families.Binomial()})

df2 = imp_data.next_sample()


# for j in range(10):
    # imp_data.update_all()


print('hello')
print(df2)
print('bye')

df2.to_csv('archive/our_testing.csv')
    # Uncomment this
