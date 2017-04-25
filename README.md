# dlmPython
DLMs in Python see West &amp; Harrison 1997


# Install

To install, the following command in a terminal:

```bash
pip install git+git://github.com/luiarthur/dlmPython
```

After the first install, the library can be updated by using the
previous command and appending `--upgrade`, as follows:

```bash
pip install git+git://github.com/luiarthur/dlmPython --upgrade
```


# Demo

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from dlmPython import dlm_uni_df, lego, param

np.random.seed(2404)

### Generate data ###

# n: number of observations
n = 20

# nAhead: number of observations to forecast
nAhead = 10

# idx: create some arbitrary time indices for the data
idx = np.linspace(1, n, n)

# y: observations that are time-dependent (linearly)
y = idx + np.random.normal(10, 1, n)

# plot data (just for fun)
plt.plot(idx, y, 'm*')
plt.show()


### Create DLM Object ###

# DLM with Linear Trend
c = dlm_uni_df(
        F=lego.E2(2), G=lego.J(2), delta=.95)

# Initialize DLM

# The state vector has a prior mean of 0, which is in fact quite far away from
# the true value of the first observation (around 10).
# The state vector has a prior covariance matrix of an Identity matrix.
init = param.uni_df(
        m=np.asmatrix(np.zeros((2,1))), 
        C=np.eye(2))

### Feed data to Kalman-filter ###
# Parameters and predictions are now begin made sequentially
filt = c.filter(y, init)

### One-step-ahead predictions at each time step ###
# predictions are distributed as student-t distributions with parameters:
#   f: center of prediction of next observation
#   Q: scale of prediction of next observation
#   n: the degrees of freedom + 1
one_step_f = map(lambda x: x.f, filt)
one_step_Q = map(lambda x: x.Q, filt)
one_step_n = map(lambda x: x.n, filt)
# credible intervals for predictions
ci_one_step = c.get_ci(one_step_f, one_step_Q, one_step_n)

### Forecasts ###
# Forecasts of observations some number of 
# time steps ahead. In this example, we have 20 observations, and
# we are making forecasts of 10 time-steps ahead. (Note: nAhead=10)
# This is different from one-step-ahead predictions only in that
# the one-step-ahead predictions are made for one time-step in the future.
fc = c.forecast(filt, nAhead)
future_idx = np.linspace(n+1, n+nAhead, nAhead)
fc_f = fc['f']
fc_Q = fc['Q']
fc_n = [fc['n']] * nAhead
ci = c.get_ci(fc_f, fc_Q, fc_n)

### Plot Results ###
# Notice that since the prior mean is far away from the first observation, the
# one-step ahead predictions are bad at the beginning.

plt.fill_between(idx, ci_one_step['lower'], ci_one_step['upper'], color='lightblue')
plt.fill_between(future_idx, ci['lower'], ci['upper'], color='pink')

plt.scatter(idx, y, color='grey', label='Data', s=15)
plt.plot(future_idx, fc_f, 'r--', label='Forecast')
plt.plot(idx, one_step_f, 'b--', label='One-step-Ahead')

plt.xlabel('time')
legend = plt.legend(loc='lower right')
plt.show()
```


# Demo of 2-nd Order Polynomial Trend

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from dlmPython import dlm_uni_df, lego, param

np.random.seed(2404)

### Generate data ###

n = 20
nAhead = 10
idx = np.linspace(1, n, n)
y = idx**2 + np.random.normal(10, 1, n)

# plot data (just for fun)
plt.plot(idx, y, 'm*')
plt.show()


### Create DLM Object ###
# DLM with Quadratic Trend
# This is the only part that is different!!! 
# (F,G) define the forecast function. In this case the 
# forecast function will be quadratic in time.
c = dlm_uni_df(
        F=lego.E2(3), G=lego.J(3), delta=.95)

# Initialize DLM
init = param.uni_df(
        m=np.asmatrix(np.zeros((3,1))), 
        C=np.eye(3))

### Feed data to Kalman-filter ###
filt = c.filter(y, init)

### One-step-ahead predictions at each time step ###
one_step_f = map(lambda x: x.f, filt)
one_step_Q = map(lambda x: x.Q, filt)
one_step_n = map(lambda x: x.n, filt)
# credible intervals for predictions
ci_one_step = c.get_ci(one_step_f, one_step_Q, one_step_n)

### Forecasts ###
fc = c.forecast(filt, nAhead)
future_idx = np.linspace(n+1, n+nAhead, nAhead)
fc_f = fc['f']
fc_Q = fc['Q']
fc_n = [fc['n']] * nAhead
ci = c.get_ci(fc_f, fc_Q, fc_n)

### Plot Results ###
# Notice that since the prior mean is far away from the first observation, the
# one-step ahead predictions are bad at the beginning.

plt.fill_between(idx, ci_one_step['lower'], ci_one_step['upper'], color='lightblue')
plt.fill_between(future_idx, ci['lower'], ci['upper'], color='pink')

plt.scatter(idx, y, color='grey', label='Data', s=15)
plt.plot(future_idx, fc_f, 'r--', label='Forecast')
plt.plot(idx, one_step_f, 'b--', label='One-step-Ahead')

plt.xlabel('time')
legend = plt.legend(loc='lower right')
plt.show()
```



# Info

This package was structured following python's packaging
guidelines[here][1].


[1]: https://python-packaging.readthedocs.io/en/latest/index.html
