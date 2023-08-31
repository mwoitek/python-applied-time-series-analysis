# %% [markdown]
# # ARMA

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

warnings.filterwarnings("ignore")

# %%
plt.rcParams["figure.figsize"] = [10, 7.5]

# %% [markdown]
# ## Simulate ARMA(1, 1)

# %%
ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])

# %%
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

# %%
plt.plot(ARMA_1)
plt.title("Simulated ARMA(1, 1) Process")
plt.xlim(0, 200)

# %%
plot_acf(ARMA_1, auto_ylims=True);  # fmt: skip

# %%
plot_pacf(ARMA_1, auto_ylims=True);  # fmt: skip

# %% [markdown]
# ## Simulate ARMA(2, 2)

# %%
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])

# %%
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)

# %%
plt.plot(ARMA_2)
plt.title("Simulated ARMA(2, 2) Process")
plt.xlim(0, 200)

# %%
plot_acf(ARMA_2, auto_ylims=True);  # fmt: skip

# %%
plot_pacf(ARMA_2, auto_ylims=True);  # fmt: skip
