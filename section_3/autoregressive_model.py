# %% [markdown]
# # Autoregressive Model

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller

# %%
plt.rcParams["figure.figsize"] = [10, 7.5]

# %% [markdown]
# ## Simulate AR(2) Process
#
# $y_t = 0.33 y_{t-1} + 0.5 y_{t-2}$

# %%
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0, 0])

# %%
AR2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# %%
plt.plot(AR2_process)
plt.title("Simulated AR(2) Process")
plt.xlim(0, 200)

# %%
plot_acf(AR2_process, auto_ylims=True);  # fmt: skip

# %%
plot_pacf(AR2_process, auto_ylims=True);  # fmt: skip

# %%
rho, sigma = yule_walker(AR2_process, 2, method="mle")
print(f"rho: {-rho}")
print(f"sigma: {sigma}")

# %% [markdown]
# ## Simulate AR(3) Process
#
# $y_t = 0.33 y_{t-1} + 0.5 y_{t-2} + 0.07 y_{t-3}$

# %%
ar3 = np.array([1, 0.33, 0.5, 0.07])
ma3 = np.array([1, 0, 0, 0])

# %%
AR3_process = ArmaProcess(ar3, ma3).generate_sample(nsample=10000)

# %%
plt.plot(AR3_process)
plt.title("Simulated AR(3) Process")
plt.xlim(0, 200)

# %%
plot_acf(AR3_process, auto_ylims=True);  # fmt: skip

# %%
plot_pacf(AR3_process, auto_ylims=True);  # fmt: skip

# %%
rho, sigma = yule_walker(AR3_process, 3, method="mle")
print(f"rho: {-rho}")
print(f"sigma: {sigma}")

# %% [markdown]
# # Mini Project: Model Johnson&Johnson Quarterly Earnings per Share (EPS)

# %%
data = pd.read_csv("jj.csv")
data.head()

# %%
plt.figure(figsize=(15, 7.5))
plt.scatter(data["date"], data["data"])
plt.title("Quarterly EPS for J&J")
plt.xlabel("Date")
plt.ylabel("EPS ($)")
plt.xticks(rotation=90)

# %%
# Take the log difference
data["data"] = np.log(data["data"])
data["data"] = data["data"].diff()
data = data.drop(data.index[0])
data.head()

# %%
plt.plot(data["data"])
plt.title("Log Difference of Quarterly EPS for J&J")

# %%
ad_fuller_result = adfuller(data["data"])
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

# %%
plot_acf(data["data"], auto_ylims=True);  # fmt: skip

# %%
plot_pacf(data["data"], auto_ylims=True);  # fmt: skip

# %%
# Try AR(4)
rho, sigma = yule_walker(data["data"], 4, method="mle")
print(f"rho: {-rho}")
print(f"sigma: {sigma}")
