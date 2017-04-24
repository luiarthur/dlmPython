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
from dlmPython import dlm_uni_df, lego, param
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


c = dlm_uni_df(
        F=lego.E2(2), G=lego.J(2), V=1, 
        delta=.95)
n = 20
nAhead = 20
idx = np.linspace(1, n, n)

y = idx + np.random.normal(0, 1, n)
init = param.uni_df(
        m=np.asmatrix(np.zeros((2,1))), 
        C=np.eye(2))
filt = c.filter(y, init)

fc = c.forecast(filt, nAhead)
future_idx = np.linspace(n+1, n+nAhead, nAhead)
fc_f = fc['f']
fc_Q = fc['Q']
fc_n = fc['n']
ci = c.get_ci(fc_f, fc_Q, fc_n)   

plt.plot(idx, y, 'ro', future_idx, fc_f, '--')
plt.plot(future_idx, ci['lower'], '--')
plt.plot(future_idx, ci['upper'], '--')
plt.xlabel('time')
plt.show()

### TODO ###
### Compare intervals with dlmScala

```



# Info

This package was structured following python's packaging
guidelines[here][1].


[1]: https://python-packaging.readthedocs.io/en/latest/index.html
