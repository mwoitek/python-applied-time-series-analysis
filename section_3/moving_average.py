# %% [markdown]
# # Moving Average Process

# %%
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

# %%
plt.rcParams["figure.figsize"] = [10, 7.5]

# %% [markdown]
# ## Simulate MA(2) Process
#
# $y_t = 0.9 Z_{t-1} + 0.3 Z_{t-2}$

# %%
ma2 = np.array([1, 0.9, 0.3])
ar2 = np.array([1, 0, 0])

# %%
MA2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# %%
plt.plot(MA2_process)
plt.title("Moving Average Process of Order 2")
plt.show()

# %%
plt.plot(MA2_process)
plt.title("Moving Average Process of Order 2")
plt.xlim(0, 200)
plt.show()

# %%
plot_acf(MA2_process, lags=20, auto_ylims=True)

# %%
MA_model = ARIMA(
    MA2_process,
    order=(0, 0, 2),
    enforce_stationarity=False,
).fit()
MA_model.summary()
