# %% [markdown]
# # Random Walk

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

# %%
plt.rcParams["figure.figsize"] = [10, 7.5]

# %% [markdown]
# ## Simulation of Random Walk

# %%
steps = np.random.standard_normal(1000)
steps[0] = 0

random_walk = np.cumsum(steps)

# %%
random_walk[:10]

# %%
plt.plot(random_walk)
plt.title("Simulated Random Walk")

# %%
random_walk_acf_coef = acf(random_walk)
random_walk_acf_coef

# %%
plot_acf(random_walk, lags=20, auto_ylims=True);  # fmt: skip

# %%
random_walk_diff = np.diff(random_walk, n=1)

# %%
plt.plot(random_walk_diff)
plt.title("Noise")

# %%
plot_acf(random_walk_diff, lags=20, auto_ylims=True);  # fmt: skip
